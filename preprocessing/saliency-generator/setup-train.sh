# Create all the required folders
mkdir storage
mkdir storage/weights

# Install all required packages
pip3 install -r requirements.txt

# Download all the training files
python3 downloader.py --type train

# Extract all required folders and remove the zip files
unzip storage/dataset.zip -d storage/
rm storage/dataset.zip
