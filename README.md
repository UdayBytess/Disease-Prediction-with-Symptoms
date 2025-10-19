# Disease Prediction from Symptoms Using Machine Learning

Disease Prediction from Symptoms involves using machine learning (ML) techniques to identify and compare medical cases based on clinical features such as symptoms, diagnoses, test results, and treatment outcomes. This approach helps healthcare professionals find similar cases, aiding in diagnosis, treatment recommendations, and patient management.

## How It Works:

Data Collection: 
Medical records, including structured (e.g., lab results) and unstructured data (e.g., clinical notes).

Data Preprocessing: 
Cleaning, normalizing, and encoding data into a machine-readable format.

Feature Extraction: 
Extracting relevant features like patient demographics, vital signs, lab results, and clinical histories.

Similarity Measurement: 
Using algorithms such as cosine similarity, Euclidean distance, or more advanced techniques like neural embeddings to measure case similarity.

Model Training: 
Training ML models (e.g., k-NN, clustering algorithms, deep learning models) to learn patterns from labeled or unlabeled patient data.

Evaluation & Deployment: 
Evaluating model performance using metrics like precision, recall, and F1-score, then deploying in a healthcare application.



## Applications:

Clinical Decision Support: Recommending diagnoses or treatments based on similar past cases.

Medical Research: Identifying cohorts for clinical trials.

Patient Risk Prediction: Assessing the likelihood of complications by comparing similar patient histories.

Personalized Medicine: Tailoring treatments based on individual case similarities.



## Challenges:

Data Privacy and Security: Ensuring compliance with healthcare data protection standards like HIPAA.

Data Quality: Dealing with incomplete or inconsistent records.

Interpretability: Making model predictions understandable for clinical use.

Ethical Considerations: Avoiding biases that may arise from imbalanced datasets.

By leveraging Disease Prediction from Symptoms using machine learning, healthcare providers can enhance clinical decision-making, improve patient outcomes, and streamline medical research processes.


## Installation
1. Install the Anaconda Python Package and TensorFlow 2.10 
	Follow this video to understand, how to Install Anaconda and TensorFlow 2.10
	https://youtu.be/b9e3J-NJ8TY
	
	Important Note:
	If TensorFlow is not running properly, downgrade the numpy version to 1.26.4 using the below command
	>> pip install numpy==1.26.4
	
2. Open anaconda prompt (Search for Anaconda prompt and open) 

3. Change the directory to the project folder using the below command
	>> cd path_of_project_folder
	Example: cd D:\Disease_Prediction_From_Symptoms
	
4. Activate the virtual environment created while installing TensorFlow using the command
	>> conda activate tf 
	
	Note: Here tf is the virtual environment name created while installing TensorFlow
	
5. Now install the required libraries using the below command
	>> pip install -r requirements.txt
	

## Follow the steps to train and run the project after installing the requirements.

1. Open anaconda prompt (Search for Anaconda prompt and open) 

2. Change the directory to the project folder using the below command
	>> cd path_of_project_folder
	Example: cd D:\Disease_Prediction_From_Symptoms
	
3. Activate the virtual environment created while installing TensorFlow using the command
	>> conda activate tf 

4. Next to train the model open the Jupyter Notebook using the below command
	>> jupyter notebook
	
5. Open the disease_prediction_model.ipynb and run all cells

6. Once the training is completed the trained model that is disease_prediction_model.h5 and preprocessing that is preprocessing.pkl will be stored in the current working directory

7. To run the Flask app open the Anaconda prompt and type the following command
	>> python app.py
	
	
## Follow the steps to run the project after installing the requirements and Training the Model.

1. Open anaconda prompt (Search for Anaconda prompt and open) 

2. Change the directory to the project folder using the below command
	>> cd path_of_project_folder
	Example: cd D:\Disease_Prediction_From_Symptoms
	
3. Activate the virtual environment created while installing TensorFlow using the command
	>> conda activate tf 

4. To run the Flask app open the Anaconda prompt and type the following command
	>> python app.py 

Happy Learning

Team VTUPulse.com
