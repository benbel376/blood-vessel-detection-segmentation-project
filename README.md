# Retina Blood Vessel Segmentation

This project implements a deep learning-based U-Net model for segmenting blood vessels in retina images. This segmentation is crucial in diagnosing and monitoring various eye-related diseases, including diabetic retinopathy. The project leverages a custom U-Net architecture and serves predictions via a FastAPI interface, making it accessible for real-time inference. The application is containerized using Docker, with NGINX set up as a reverse proxy for deployment.

## Table of Contents
- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Requirements](#requirements)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Future Improvements](#future-improvements)
- [License](#license)

## Project Overview
The primary goal of this project is to provide a reliable tool for the segmentation of blood vessels in retina images using a deep learning model. The U-Net model is trained on retina image datasets and evaluated for performance on key metrics such as Jaccard index, F1 score, and accuracy. The trained model is then served through a FastAPI application, which can be accessed via a RESTful API for easy integration into other healthcare applications.

## Folder Structure
Retina Blood Vessel Segmentation Project 
├── notebooks/ # Jupyter notebooks for data exploration and experiments 
├── src/ 
│ ├── app/ 
│ │ ├── app.py # FastAPI application code 
│ │ ├── dataset.py # Dataset class for loading and processing images 
│ │ ├── model.py # U-Net model architecture 
│ │ ├── losses.py # Custom loss functions for training 
│ │ ├── train.py # Script for training the model 
│ │ ├── utils.py # Utility functions (e.g., metrics calculation) 
│ │ ├── main.py # Main execution script 
│ │ ├── requirements.txt # Python dependencies 
│ │ └── Dockerfile # Docker configuration for the FastAPI app 
│ └── infra/ 
│ ├── docker-compose.yml # Docker Compose setup 
│ ├── nginx.conf # NGINX configuration for reverse proxy 
│ ├── run.sh # Script to run the setup and deployment 
│ └── README.md # Documentation for infrastructure setup



## Requirements
- Linux
- Python 3.8 or higher
- CUDA-enabled GPU (optional for faster training)
- Docker
- Docker Compose
- FastAPI
- NGINX
- PyTorch

## Setup and Installation

1. **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/retina-blood-vessel-segmentation.git
    cd retina-blood-vessel-segmentation
    ```

2. **Install Python Dependencies**
    Make sure to create a virtual environment and activate it before installing dependencies.
    ```bash
    cd src/app
    pip install -r requirements.txt
    ```

3. **Build Docker Containers**
    From the root directory, run Docker Compose to build and start the containers.
    ```bash
    docker-compose -f src/infra/docker-compose.yml up --build
    ```

4. **Configure NGINX (Optional)**
   If NGINX is not already installed, follow the instructions in `nginx.conf` for reverse proxy setup.

5. **Run the FastAPI Application**
   After setting up Docker, you can start the FastAPI app using:
    ```bash
    sh src/infra/run.sh
    ```

## Usage

### Accessing the FastAPI Application
Once the application is running, access it at `http://localhost:8000/docs` to view and interact with the FastAPI Swagger UI. Here, you can send images of retina scans to the model and receive segmented output in real-time.

### Training the Model
To train the model, modify the `train.py` parameters as desired and run:
```bash
python src/app/train.py
```

## Future Improvements
- Data Augmentation: Implement additional data augmentation techniques to enhance model robustness.
- Model Optimization: Experiment with more efficient architectures like U-Net++ or attention U-Net for improved accuracy.
- Real-Time GPU Inference: Set up GPU-based inference for faster processing times in production.
- Cloud Deployment: Deploy the application on cloud platforms like AWS or GCP for wider accessibility.

## License
This project is licensed under the MIT License. See the LICENSE file for details.


This README provides a clear overview of the project, installation instructions, and details about each folder and file. Let me know if you want any further customization!
