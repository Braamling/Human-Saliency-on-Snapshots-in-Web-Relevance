## Two-Stage Transfer Learning of End-to-End Convolutional Neural Networks for Webpage Saliency Prediction
Implementation by *Bram van den Akker*.
Original paper by *Shan, Wei and Sun, Guangling and Zhou, Xiaofei and Liu, Zhi*

**Note: Before starting, make sure unzip, python3 and pip3 are installed.**
#### Setting up saliency predictor.
Depending on your usage, there are several setups possible. Please select the one that is applicable for your use case.
##### Inference setup (recommended)
If your application only requires a pre-trained inference model, the inference setup should be enough. This setup will install the requirements, setup the folder structure and download the pre-trained model weights. All required steps can be performed using:
>> sh setup-infer.sh
##### Training setup (less recommended)
If you want to retrain an existing model or create a new model, the training setup is required. This will download all the processed images and heatmaps from salicon and FiWi datasets, create the required folders and install all python packages. All required steps can be performed using:
>> sh setup-train.sh
##### Full setup (not recommended)
If you want to change anything to the preprocessing of the Salicon and FiWi datasets (ie. gaussian episilon or using it raw) you will need to run the full setup. Note: this setup takes quite a while. This will download the raw Salicon and FiWi dataset, install the salicon and msCOCO libaries, creates the heatmaps, makes the FiWi training/validation split, create the required folders and install all python packages.
>> sh setup-full.sh
##### Manual setup (not recommended)
TODO, but for now please check the content of the bash scripts above.

1) Install the coco api using the setup.py in /lib/cocoapi/PythonAPI.
2) Install the salicon api using the setup.py in /lib/salicon

Credits:
SALICON-api has been forked from https://github.com/NUS-VIP/salicon-api and is modified to work with python3.
COCO-api has been forked from https://github.com/cocodataset/cocoapi
Salicon-evaluation has been forked from https://github.com/NUS-VIP/salicon-evaluation
Drive-download code has been taken from Stackoverflow user `turdus-merula` on
https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
>@inproceedings{shan2017two,
>  title={Two-Stage Transfer Learning of End-to-End Convolutional Neural Networks for Webpage Saliency Prediction},
>  author={Shan, Wei and Sun, Guangling and Zhou, Xiaofei and Liu, Zhi},
>  booktitle={International Conference on Intelligent Science and Big Data Engineering},
>  pages={316--324},
>  year={2017},
>  organization={Springer}
>}