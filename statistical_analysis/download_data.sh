#!/bin/bash

# Google Drive Multi-Folder Downloader
# Downloads multiple folders in parallel from Google Drive links

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Hardcoded Google Drive folder URLs (add more as needed)
GDRIVE_URLS=(
    "https://drive.google.com/drive/folders/1AMGiKz185ZDMoilXVPLgohjzzhusJW__?usp=sharing"
    "https://drive.google.com/drive/folders/1pe9HB7sfowRps9wvFjL7JEldkTVrO2Hq?usp=sharing"
)

# Extract folder ID from URL
extract_folder_id() {
    local url=$1
    if [[ $url =~ folders/([a-zA-Z0-9_-]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    fi
}

# Get folder name from Google Drive
get_folder_name() {
    local folder_id=$1
    local name=$(rclone lsjson --drive-root-folder-id="$folder_id" gdrive: -d 0 2>/dev/null | grep -o '"Name":"[^"]*"' | head -1 | cut -d'"' -f4)
    
    if [ -z "$name" ]; then
        name=$(rclone lsf --drive-root-folder-id="$folder_id" gdrive: --dirs-only -d 0 2>/dev/null | tr -d '/')
    fi
    
    if [ -z "$name" ]; then
        name="$folder_id"
    fi
    
    echo "$name"
}

# Download a single folder (called as background process)
download_folder() {
    local url=$1
    local output_base=$2
    local index=$3
    
    local folder_id=$(extract_folder_id "$url")
    
    if [ -z "$folder_id" ]; then
        echo -e "${RED}[$index] Error: Could not extract folder ID from: $url${NC}"
        return 1
    fi
    
    local folder_name=$(get_folder_name "$folder_id")
    local output_dir="$output_base/$folder_name"
    mkdir -p "$output_dir"
    
    echo -e "${BLUE}[$index] Starting download: $folder_name${NC}"
    echo -e "${YELLOW}[$index] Folder ID: $folder_id${NC}"
    echo -e "${YELLOW}[$index] Saving to: $output_dir${NC}"
    
    rclone copy --drive-root-folder-id="$folder_id" gdrive: "$output_dir" \
        --progress \
        --transfers=4 \
        --checkers=8 \
        --stats-one-line \
        --stats 5s \
        --log-file="$output_base/.rclone_log_$index.txt"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[$index] Completed: $folder_name${NC}"
    else
        echo -e "${RED}[$index] Failed: $folder_name${NC}"
        return 1
    fi
}

main() {
    echo -e "${GREEN}=== Google Drive Multi-Folder Downloader ===${NC}"
    
    local output_base="${1:-.}"
    
    # Check if rclone is configured
    if ! rclone listremotes | grep -q "gdrive:"; then
        echo -e "${RED}Error: gdrive remote not configured. Run 'rclone config' first.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Found ${#GDRIVE_URLS[@]} folders to download${NC}"
    echo -e "${YELLOW}Output directory: $output_base${NC}"
    echo ""
    
    # Array to store background process PIDs
    pids=()
    
    # Start downloads in parallel
    for i in "${!GDRIVE_URLS[@]}"; do
        download_folder "${GDRIVE_URLS[$i]}" "$output_base" "$((i+1))" &
        pids+=($!)
        sleep 1  # Small delay to avoid rate limiting
    done
    
    echo -e "${GREEN}All downloads started in parallel...${NC}"
    echo ""
    
    # Wait for all downloads to complete
    failed=0
    for i in "${!pids[@]}"; do
        if ! wait ${pids[$i]}; then
            ((failed++))
        fi
    done
    
    echo ""
    echo -e "${GREEN}=== All Downloads Complete ===${NC}"
    
    if [ $failed -gt 0 ]; then
        echo -e "${RED}$failed download(s) failed. Check log files for details.${NC}"
    else
        echo -e "${GREEN}All folders downloaded successfully!${NC}"
    fi
    
    # List downloaded folders
    echo -e "${YELLOW}Downloaded folders:${NC}"
    find "$output_base" -maxdepth 1 -type d -not -path "$output_base" -exec basename {} \; | sort
}

main "$@"
