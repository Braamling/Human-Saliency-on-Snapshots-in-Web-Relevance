# Create all the required folders
mkdir storage
mkdir storage/weights

# Install all required packages
pip3 install -r requirements.txt

# Download all the training files
python3 downloader.py --type train

# Extract all required folders and remove the zip files
unzip storage/salicon/val_images.zip -d storage/salicon/images

rm storage/salicon/val_images.zip
rm storage/salicon/train_images.zip
rm storage/FiWi/dataset.zip

# Create the fixation data.
python3 createFixationImages.py