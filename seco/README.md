## SECO train and test

### Datasets

In lift-the-flap task, we use images from COCO-Stuff for pretraining, please download the dataset from [here](https://github.com/nightrome/cocostuff). We use test images from OCD and PASCAL VOC2007, you can download datasets from [here](https://github.com/kreimanlab/WhenPigsFlyContext)  and [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) respectively. 

We propose the **h**uman **o**bject **p**riming dataset (HOP), which can be found in [this folder](https://drive.google.com/drive/folders/1HCJwgkhSN9SGCdZfrBK9pmmTMpjvHSWv?usp=sharing).

### Downloading the pretrained models and required artifacts (~10GB)

Run the following shell script to download all the pretrained models and the artifacts. The script prompts to use gdown or rclone for the download. Recommended: **rclone** because gdown has rate limits. Rclone requires you to authenticate using your Google account, which can be done using the link provided in the terminal once you run the shell script.
```
./seco_artifacts_download.sh
```
If you get authentication errors, try clearing your previous rclone config using
```
rclone config delete gdrive
```
And then try again, else choose gdown for download

Or you can directly download from [here](https://drive.google.com/drive/folders/1Lu2ABx4NqKNkhSvGLbWMZwBticQbF35C?usp=sharing)


### Main Results

#### Training

To reproduce the pre-training phase, run the following commands for different datasets:

- **COCO-OCD**:  ```nohup python main_seco.py -a resnet50 --img_dir /PATH/TO/COCO/IMAGES --anno_dir annotations --epochs 500 -b 256 --dist-url 'tcp://localhost:10012' --resume 'pre-trained_models/imagenet.pth.tar' --save_dir 'checkpoints_ocd' --multiprocessing-distributed --world-size 1 --rank 0 --dataset ocd --mlp 512 --K 200 --memory_dim 512 --workers 16 > train_ocd.log 2>&1 &```
- **COCO-VOC**:  ```nohup python main_seco.py -a resnet50 --img_dir /PATH/TO/COCO/IMAGES --anno_dir annotations --epochs 500 -b 256 --dist-url 'tcp://localhost:10012' --resume 'pre-trained_models/imagenet.pth.tar' --save_dir 'checkpoints_voc' --multiprocessing-distributed --world-size 1 --rank 0 --dataset voc --mlp 512 --K 200 --memory_dim 512 --workers 16 > train_voc.log 2>&1 &```

#### Evaluation

We have provided our pretrained models and trained linear weights for evaluation in these folders [1]([https://drive.google.com/drive/u/2/folders/1AHr_eX46uk3uSYS2CJIhRUfG_RsY1-Vv](https://drive.google.com/drive/folders/1MrEoTYzMIvodSMnhtpx7NnC5ubSWH3nD?usp=drive_link)) [2]([https://drive.google.com/drive/u/2/folders/1DmwWT3VCOkruCTs2mGSceLdD615VGB-p](https://drive.google.com/drive/folders/1hTgt301nLcf--YxcDGh76Ij8QiUsDwaI?usp=sharing)). To reproduce main results, run following commands for different datasets:

- **OCD In Domain**: ```nohup python eval_linear.py --evaluate --batch_size_per_gpu 256 --anno_dir annotations --img_train_dir /PATH/TO/COCO/TRAIN/IMAGES --img_val_dir /PATH/TO/COCO/VAL/IMAGES --dataset ocd --arch resnet50 --pretrained_weights pre-trained_models/ocd_eps500.pth.tar --linear_weights linear_weights/ocd/checkpoint_best.pth.tar  --lr 0.1 --num_labels 15 --output_dir 'eval_checkpoints_ocd' --dist_url 'tcp://localhost:10002' --gpu 0 > logs/eval_seco_ocd_indomain.log 2>&1 &```
- **OCD Out of Domain**: ```nohup python eval_linear_ocd.py --batch_size_per_gpu 256 --anno_dir /PATH/TO/OCD --img_val_dir /PATH/TO/OCD --arch resnet50 --pretrained_weights pre-trained_models/ocd_eps500.pth.tar --linear_weights linear_weights/ocd/checkpoint_best.pth.tar --num_labels 15  --dist_url 'tcp://localhost:10001' --gpu 0 > logs/eval_seco_ocd_outofdomain.log 2>&1 &```

- **VOC In Domain**: ```nohup python eval_linear.py --evaluate --batch_size_per_gpu 256 --anno_dir annotations --img_train_dir /PATH/TO/COCO/TRAIN/IMAGES --img_val_dir /PATH/TO/COCO/VAL/IMAGES --dataset voc --arch resnet50 --pretrained_weights pre-trained_models/voc_eps500.pth.tar --linear_weights linear_weights/voc/checkpoint_best.pth.tar  --lr 0.1 --num_labels 20 --output_dir 'eval_checkpoints_voc' --dist_url 'tcp://localhost:10001' --gpu 1 > logs/eval_seco_voc_indomain.log 2>&1 &```
- **VOC Out of Domain**:```nohup python eval_linear_voc07.py --batch_size_per_gpu 256 --anno_dir /PATH/TO/VOC/ANNOTATIONS  --img_val_dir /PATH/TO/VOC/VALSET --test_split_dir /PATH/TO/VOC/TESTSPLIT --arch resnet50 --pretrained_weights pre-trained_models/voc_eps500.pth.tar --linear_weights linear_weights/voc/checkpoint_best.pth.tar --num_labels 20  --gpu 2 > logs/eval_seco_voc_outofdomain.log 2>&1 &```
- **HOP**: ```nohup python eval_object_priming.py > logs/eval_seco_op.log 2>&1 &```

### Ablation Results

To reproduce all the experiments in ablation study, please run commands in this [shell file](https://drive.google.com/drive/u/2/folders/1hz6u-PH2IleEM3Sh2TW1mkd8qsadJEcg) separately.

```
./eval_linear.sh
```

### Logs

Alternatively, you can refer to logs for all experiments mentioned above:

- **Training**: [COCO-OCD](https://drive.google.com/file/d/150q9OjhmfjC_hMbdbpixl8P2nmY2gZh6/view?usp=sharing),  [COCO-VOC](https://drive.google.com/file/d/1Z15HnDmJczCDdrghr2zVVaWCxZniiZd3/view?usp=sharing)
- **Evaluation**: 
  - COCO-OCD: [In Domain](https://drive.google.com/file/d/1jrkS8I0ugnv9y_O4gSpacd-U7IFs92hz/view?usp=sharing), [Out of Domain](https://drive.google.com/file/d/1fn25qoDhUyU3WRHnXywS0AN9lyKghwfI/view?usp=sharing)
  - COCO-VOC: [In Domain](https://drive.google.com/file/d/1k0EdbEQdmbskh_AKBw16lVwvOgnV2zcp/view?usp=sharing), [Out of Domain](https://drive.google.com/file/d/1BKy47ark-IQzjxOaN_ieBr6Yf7zAEWrT/view?usp=sharing)
  - Object Priming: [HOP](https://drive.google.com/file/d/1nIlCSGTwfRtfkJ4Bc562gC6xinaNLDtx/view?usp=sharing)
- **Ablations**: 
  - Ablations: [Ablation Logs](https://drive.google.com/drive/folders/1tqde5Ed6poI8aceC4VkS-fLfX6lZxcBp?usp=sharing)
  - Memory slots and dimensions: [Memory Slots](https://drive.google.com/drive/folders/1sH3EQeeHaKFxriIuqeeOcCtWtU0hbEPT?usp=sharing)
 
![Ablations](images/ablation_study.png)

![Memory slots and dimensions](images/slots_memory.png)


