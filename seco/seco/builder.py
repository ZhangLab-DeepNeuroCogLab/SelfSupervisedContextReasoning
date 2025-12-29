# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.models as models
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None, fill=True):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if fill:
            if mask is not None:
                attn = attn.masked_fill(mask == 0, -1e9)
            
            attn = self.dropout(F.softmax(attn, dim=-1))
        else:
            attn = attn * mask
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        # self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    def forward(self, q, k, v, mask=None, fill=True):
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v
        bs_q, K_memory = q.size(0), k.size(0)
        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        # unsequeeze to make query length of 1
        q = q.view(bs_q, 1, n_head, d_k) 
        # unsequeeze to have a fake batch size of 1
        k = self.w_ks(k).view(1, K_memory, n_head, d_k)
        # v = self.w_vs(v).view(1, K_memory, n_head, d_v)
        v = v.view(1, K_memory, n_head, d_v)
        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)


        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        # b x n x 1 x dq 
        q, attn = self.attention(q, k, v, mask=mask, fill=fill)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(bs_q, -1)
        q = self.dropout(self.fc(q))
        # q += residual

        q = self.layer_norm(q)

        return q, attn

class InfoLoss(nn.Module):
    # from VICReg: https://arxiv.org/abs/2105.04906
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        dim = x.size(1)
        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        batch_size = x.shape[0]
        x = x - x.mean(dim=0)
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x))
        cov_x = (x.T @ x) / (batch_size - 1)        
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div(dim)
        return std_loss, cov_loss

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class SeCoLoss(nn.Module):
    def __init__(self, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
        super().__init__()

        self.repr_loss_crit = F.mse_loss 
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.info_loss = InfoLoss()

    def forward(self, h_x, h_y):
        repr_loss = self.repr_loss_crit(h_x, h_y)

        x_std, x_cov = self.info_loss(h_x)
        y_std, y_cov = self.info_loss(h_y)

        std_loss = (x_std + y_std) /2
        cov_loss = x_cov + y_cov

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        loss_scalars = [
            loss.item(), repr_loss.item(),
            x_std.item(), x_cov.item(), 
            y_std.item(), y_cov.item()
        ]

        return loss, loss_scalars


class SeCo(nn.Module):

    def __init__(self, args): 

        super().__init__()
        self.args = args
        self.num_features = int(args.mlp.split("-")[-1])
        self.backbone_context = models.__dict__[args.arch](
            zero_init_residual=True
        )
        self.embedding = self.backbone_context.fc.weight.shape[1]
        self.backbone_context.fc = nn.Identity()

        self.backbone_object = models.__dict__[args.arch](
            zero_init_residual=True
        )
        self.embedding = self.backbone_object.fc.weight.shape[1]
        self.backbone_object.fc = nn.Identity()

        self.projector_1 = Projector(args, self.embedding)
        self.projector = Projector(args, self.embedding)

        # build an external memory
        self.memory_dim = args.memory_dim
        self.K = args.K
        self.n_head = args.memory_nhead
        self.ext_memory = nn.Parameter(torch.zeros(self.K, self.memory_dim))
        nn.init.xavier_uniform_(self.ext_memory)
        self.attention = MultiHeadAttention(self.n_head, 
                                            self.memory_dim, 
                                            self.memory_dim//self.n_head, 
                                            self.memory_dim//self.n_head)

    def retrieve_memory(self, x, external_repr=None):
        if external_repr is None:
            x = backbone_context(x)
        else:
            x = external_repr
        q = self.projector_1(x)
        m, attn = self.attention(q, self.ext_memory, self.ext_memory)
        return m, attn 

    def forward(self, masked_context, objects):
 
        x = self.backbone_context(masked_context)
        y = self.backbone_object(objects)

        q = self.projector_1(x)
        r = self.projector(y) # without external memory

        m,_ = self.attention(q, self.ext_memory, self.ext_memory)

        h_x = m 
        h_y = r

        return x, y, m, h_x, h_y

class SeCoWithLoss(nn.Module):
    def __init__(self, args):
        super().__init__()    
        self.K = args.K
        self.memory_dim = args.memory_dim
        self.n_head = args.memory_nhead
        self.seco = SeCo(args)
        self.sim_coeff = args.sim_coeff
        self.std_coeff = args.std_coeff
        self.cov_coeff = args.cov_coeff
        self.crit = SeCoLoss(self.sim_coeff, self.std_coeff, self.cov_coeff)
    
    @torch.no_grad()
    def encode(self, masked_context, objects, external_repr=None, return_attn=False):
        x = masked_context
        m, attn = self.seco.retrieve_memory(x,external_repr)
        h_context = self.seco.backbone_context(masked_context)
        h_object = self.seco.backbone_object(objects)
        h_x = h_context
        h_y = h_object
        if return_attn:
            return h_x, h_y, m, attn
        else:
            return h_x, h_y, m


    def forward(self, masked_context, objects):
        x, y, m, h_x, h_y = self.seco(masked_context, objects)
        x_metrics = self.collapse_metric(x)
        y_metrics = self.collapse_metric(y)
        m_metrics = self.collapse_metric(m)
        loss, loss_scalars = self.crit(h_x, h_y)
        return loss, loss_scalars, x_metrics, y_metrics, m_metrics
    
    @torch.no_grad()
    def collapse_metric(self, x):
        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        z = F.normalize(x,dim=1) # Z [B x C]
        # per-channel std in simsiam: https://arxiv.org/pdf/2011.10566.pdf
        # then average over channels
        std = torch.sqrt(z.var(dim=0) + 0.0001).mean(0) # [C]
        # metrics proposed in https://arxiv.org/abs/2203.16262
        o_z = z.mean(0) # over a batch [C]
        r = z - o_z # over a batch [B x C]
        # print(z.shape, o_z.shape, r.shape)
        m_r = torch.norm(r,p=2) / torch.norm(z,p=2) # over channels
        m_o = torch.norm(o_z,p=2) / torch.norm(z,p=2) # over channels

        return (std.item(), m_r.item(), m_o.item())

def Projector(args, embedding):
    mlp_spec = f"{embedding}-{args.mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


def exclude_bias_and_norm(p):
    return p.ndim == 1


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    
    from VICReg: https://arxiv.org/abs/2105.04906
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]