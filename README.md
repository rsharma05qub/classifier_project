## End to End Classifier Project using Radar Backscattered Data

This project uses deep learning (TensorFlow) to classify objects (e.g., grenade, gun, knife, scissor) using backscattered radar data. The project includes a full machine learning
pipeline, from data ingestion and preprocessing to model training and prediction. 

classifier_project/
│
├── data/               # Dataset folder
│   ├── raw/            # Original .mat files
│   └── processed/      # Transformed datasets
│
├── models/             # Saved trained models
├── logs/               # Training logs
│
├── src/                # Source code
│   ├── components/     # Core pipeline modules
│   │   ├── data_ingestion.py
│   │   ├── model_trainer.py
│   ├── pipeline/       # Execution pipelines
│   │   ├── train_pipeline.py
│   │   ├── predict_pipeline.py
│   ├── utils.py        # Helper functions
│   ├── logger.py       # Logging utility
│   └── exception.py    # Custom exception handler
│
├── requirements.txt
├── setup.py
└── README.md           

Model Architecture:
Input shape: (441, 201, 6) - a 3D tensor from g_SR_all_z
Model type: CNN-based classifier (TensorFlow/Keras)
Output classes: Grenade, Gun, Knife, Scissor

Dataset:
Collected from synthetic radar backscatter simulations. 
Each .mat file contains:
	g_SR_all_z: complex backscattered signals 
Labels are inferred from filenames (e.g., Gun_01_gSR.mat -> class = Gun)

How to train the model:_
	python src/pipeline/train_pipeline.py
This will:
	1. Load and preprocess data.
	2. Train the model.
	3. Save it to the models/ directory. 
	
How to predict:
	python src/pipeline/predict_pipeline.py

predict_pipeline.py can be modified to accept .npy input or integrate with Streamlit/FastAPI.

This project includes a FastAPI-based API to classify backscatter radar samples. 

Endpoints: 
	1. /predict/: Predicts class from raw input array (JSON).
	2. /upload/: Uploads a .mat or .npy file containing the input sample and returns the predicted class.

Running the API:
	Make sure to install the dependencies: pip install fastapi uvicorn numpy h5py python-multipart
	Then launch the API: uvicorn main:app --reload
	Visit Swagger UI: http://127.0.0.1.8000/docs