Rebar Detection Project by HPX and JER

Overview

The Rebar Detection Project is an AI-based system that utilizes YOLO (You Only Look Once) Ultralytics for detecting and counting rebars in an image or video feed. The project is designed to automate rebar detection using deep learning, making it useful for construction and manufacturing industries.

Prerequisites

Before setting up the project, ensure that you have the following installed:

Anaconda (For managing Python environments)

Python 3.8+ Prefer, Python 3.10.9

YOLO Ultralytics

Installation Guide

Step 1: Install Anaconda

Download and install Anaconda from the official website: Anaconda Download

Step 2: Create a Virtual Environment

Open Anaconda Prompt and create a virtual environment named rebar_env:

conda create --name rebar_env python=3.10.9
conda activate rebar_env

Step 3: Install YOLO Ultralytics

Once inside the virtual environment, install YOLO Ultralytics:
in the terminal type:
conda install conda-forge::ultralytics
and PyTorch 
conda install pytorch::pytorch
and OpenCV
conda install conda-forge::opencv
and also Torch Vision 
conda install -c pytorch torchvision


Step 4: Clone the Repository

Download the project code from GitHub:

git clone https://github.com/Baboyeatami/Rebars_HPXS_Project.git

Step 5: Set Up the Project Directory

Navigate to *C:* and create a folder named Rebars_Project.

Extract the cloned repository into C:\Rebars_Project.

Inside C:\Rebars_Project, create a new folder named runs/detect.

Copy the train4 folder from the extracted repository and paste it into C:\Rebars_Project\runs\detect.

Step 6: Run the Application

Once the setup is complete, launch the Rebar Detection GUI:

cd C:\Rebars_Project
python counterGUI5.py

Project Structure

Rebars_Project/
│── runs/
│   ├── detect/
│   │   ├── train4/  # Pre-trained YOLO model
│── counterGUI5.py   # Main GUI application
│── other_project_files...

Usage

Run the counterGUI5.py script.

Load images or video streams containing rebars.

The model will detect and count the rebars automatically.

Results will be displayed in the GUI.

Contributing

If you want to contribute:

Fork the repository.

Create a feature branch.

Submit a pull request with detailed descriptions.

License

This project is licensed under the MIT License.

Contact

For any inquiries, please contact the developers:

HPX (Email: hpxsantos@gmail.com)

JER (Email: jamiewertalmighty@gmail.com)

