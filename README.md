# Nepali Heritage Forge

## Introduction
Nepali Heritage Forge is a project that aims towards digitizing sanskrit audio and images of sanskrit lines and convert them into English. 
## Goals
1) Digitize sanskrit texts from audio or images 
2) translate sanskrit to English
## Contributors
## Project Architecture


# Status
## Known Issue
Even with the available data the translator model has achieved scores as shown inside the notebook folder (epoch <epoch number>)Currently, due to lack of proper data, the system supports two to three lines of sanskrit in an image and a relatively small audio length. 
## High Level Next Steps
It can be better fine tuned with more data in future. 

# Usage
## Installation
### without using docker file
1) Create a folder named 'virtual environment'
2) now make a virtual environment named 'nepalihertiageforge' inside the virtual environment folder.
3) Activate the virtual environment
4) use pip command to install all the requirements from requirements.txt file into nepaliheritageforge.
6) copy app.py file into nepaliheritageforge
5) Due to the model, checkpoint, ffmpeg size being too large, All of their files have been uploaded to the drive link below. Download these folders into your local device and store them in the same folder containing your app.py file
drive link : https://drive.google.com/drive/folders/1LUVFxWY91p-EyANIoC-1jfo-drShnu1c?usp=sharing
6) All of the paths in app.py use the relative path notation (.\) which refers to the current working directory where the script is being run. If the script is executed in a directory containing these subdirectories (like checkpoint-90858, M2M100), the paths should work fine.
7) visit the link https://github.com/UB-Mannheim/tesseract/wiki and download the 'tesseract-ocr-w64-setup-5.4.0.20240606.exe (64 bit)' file. create a new folder named 'Tesseract-OCR' in the same folder containing the app.py file. then run the tesseract-ocr-w64-setup-5.4.0.20240606.exe and extract all the contents to Tesseract-OCR folder.
8) Then run the app.py file using streamlit
 
#### Creating Virtual Environment

This package is built using `python-3.11`. 
We recommend creating a virtual environment and using a matching version to ensure compatibility.

#### pre-commit

`pre-commit` will automatically format and lint your code. You can install using this by using
`make use-pre-commit`. It will take effect on your next `git commit`

#### pip-tools

The method of managing dependencies in this package is using `pip-tools`. To begin, run `make use-pip-tools` to install. 

Then when adding a new package requirement, update the `requirements.in` file with 
the package name. You can include a specific version if desired but it is not necessary. 

To install and use the new dependency you can run `make deps-install` or equivalently `make`

If you have other packages installed in the environment that are no longer needed, you can you `make deps-sync` to ensure that your current development environment matches the `requirements` files. 

## Usage Instructions


# Data Source
## Code Structure
## Artifacts Location

# Results
## Metrics Used
## Evaluation Results