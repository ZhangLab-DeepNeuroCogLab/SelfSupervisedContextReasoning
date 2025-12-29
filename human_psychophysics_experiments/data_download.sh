set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to extract folder ID from various Google Drive URL formats
extract_folder_id() {
    local url=$1
    local folder_id=""
    
    # Try different URL patterns
    if [[ $url =~ folders/([a-zA-Z0-9_-]+) ]]; then
        folder_id="${BASH_REMATCH[1]}"
    elif [[ $url =~ id=([a-zA-Z0-9_-]+) ]]; then
        folder_id="${BASH_REMATCH[1]}"
    elif [[ $url =~ ^[a-zA-Z0-9_-]+$ ]]; then
        # Direct folder ID provided
        folder_id=$url
    fi
    
    echo "$folder_id"
}

# Function to check and install gdown
install_gdown() {
    if ! command -v gdown &> /dev/null; then
        print_message "$YELLOW" "gdown not found. Installing..."
        
        if command -v pip3 &> /dev/null; then
            pip3 install --user gdown
        elif command -v pip &> /dev/null; then
            pip install --user gdown
        else
            print_message "$RED" "Error: pip not found. Please install Python and pip first."
            return 1
        fi
        
        # Add user pip bin to PATH if not already there
        export PATH="$HOME/.local/bin:$PATH"
        
        if command -v gdown &> /dev/null; then
            print_message "$GREEN" "gdown installed successfully!"
        else
            print_message "$RED" "Failed to install gdown"
            return 1
        fi
    else
        print_message "$GREEN" "gdown is already installed"
    fi
}

# Function to download using gdown
download_with_gdown() {
    local folder_id=$1
    local output_dir=${2:-.}
    
    print_message "$GREEN" "Downloading folder using gdown..."
    print_message "$YELLOW" "Folder ID: $folder_id"
    print_message "$YELLOW" "Output directory: $output_dir"
    
    # Create output directory if it doesn't exist
    mkdir -p "$output_dir"
    
    # Download the folder
    # Note: gdown may have limitations with large folders or folders requiring authentication
    gdown --folder "https://drive.google.com/drive/folders/$folder_id" -O "$output_dir" --remaining-ok
    
    if [ $? -eq 0 ]; then
        print_message "$GREEN" "Download completed successfully!"
    else
        print_message "$YELLOW" "Download completed with some warnings or errors"
    fi
}

# Function to setup and use rclone (alternative method)
setup_rclone() {
    if ! command -v rclone &> /dev/null; then
        print_message "$YELLOW" "rclone not found. Installing..."
        
        # Install rclone
        curl https://rclone.org/install.sh | sudo bash
        
        if [ $? -ne 0 ]; then
            print_message "$RED" "Failed to install rclone"
            return 1
        fi
    fi
    
    print_message "$GREEN" "rclone is installed"
    
    # Check if Google Drive remote is configured
    if ! rclone listremotes | grep -q "gdrive:"; then
        print_message "$YELLOW" "Setting up Google Drive remote in rclone..."
        print_message "$YELLOW" "Please follow the interactive setup:"
        rclone config create gdrive drive scope drive.readonly
    fi
}

# Function to download using rclone
download_with_rclone() {
    local folder_id=$1
    local output_dir=${2:-.}
    
    print_message "$GREEN" "Downloading folder using rclone..."
    
    # Create output directory if it doesn't exist
    mkdir -p "$output_dir"
    
    # Download - flag must come BEFORE the remote
    rclone copy --drive-root-folder-id="$folder_id" gdrive: "$output_dir" \
        --progress \
        --transfers=4 \
        --checkers=8
    
    if [ $? -eq 0 ]; then
        print_message "$GREEN" "Download completed successfully!"
    else
        print_message "$RED" "Download failed"
        return 1
    fi
}

# Main script
main() {
    print_message "$GREEN" "=== Google Drive Folder Downloader ==="
    
    # HARDCODED GOOGLE DRIVE FOLDER URL
    local gdrive_url="https://drive.google.com/drive/folders/1HCJwgkhSN9SGCdZfrBK9pmmTMpjvHSWv?usp=sharing"
    
    # Check if URL is provided as argument (optional - overrides hardcoded URL)
    if [ $# -gt 0 ]; then
        gdrive_url=$1
        local output_dir=${2:-.}
        print_message "$YELLOW" "Using provided URL instead of hardcoded one"
    else
        local output_dir=${1:-.}
        print_message "$GREEN" "Using hardcoded URL: $gdrive_url"
    fi
    
    # Extract folder ID
    local folder_id=$(extract_folder_id "$gdrive_url")
    
    if [ -z "$folder_id" ]; then
        print_message "$RED" "Error: Could not extract folder ID from URL"
        exit 1
    fi
    
    print_message "$GREEN" "Extracted folder ID: $folder_id"
    
    # Choose download method
    print_message "$YELLOW" "\nSelect download method:"
    print_message "$YELLOW" "1. Use gdown (simpler, but may have limitations with large folders)"
    print_message "$YELLOW" "2. Use rclone (more reliable, requires Google account authentication)"
    read -p "Enter your choice (1 or 2): " choice
    
    case $choice in
        1)
            install_gdown
            if [ $? -eq 0 ]; then
                download_with_gdown "$folder_id" "$output_dir"
            else
                print_message "$RED" "Failed to setup gdown"
                exit 1
            fi
            ;;
        2)
            setup_rclone
            if [ $? -eq 0 ]; then
                download_with_rclone "$folder_id" "$output_dir"
            else
                print_message "$RED" "Failed to setup rclone"
                exit 1
            fi
            ;;
        *)
            print_message "$RED" "Invalid choice"
            exit 1
            ;;
    esac
    
    print_message "$GREEN" "\n=== Download Complete ==="
    print_message "$YELLOW" "Files downloaded to: $output_dir"
    
    # List downloaded folders
    if [ -d "$output_dir" ]; then
        print_message "$YELLOW" "\nDownloaded folders:"
        find "$output_dir" -maxdepth 1 -type d -not -path "$output_dir" -exec basename {} \; | sort
    fi
}

# Run main function
main "$@"
