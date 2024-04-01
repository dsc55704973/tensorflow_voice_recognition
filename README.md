# Tensorflow Neural Network: Gender Recognition by Voice and Speech Analysis

### SUMMARY

Thank you for reviewing this project, which exhibits a neural network model I developed with Tensorflow.  The model is trained on a dataset featuring comprehensive parameters of the human voice (please review section below dedicated to describing the dataset).  Development of the neural network was straight-forward given how lean the dataset actually is (3,168 records, 20 features) and how inherently robust neural network models are.  

Essentially, I trained a model here that will predict the gender of a speaker given these 20 voice parameters.  The end result is a model that generated predictions with 98.23% accuracy, and only 14 wrong predictions!  Pretty remarkable.  The steps to execute this model are generalized as follows: 

Step 1: Exploratory Data Analysis

Step 2: Pre-Processing

Step 3: Model Development

Step 4: Quantify the Trained Model

Step 5: Make Predictions

Step 6: Evaluate Test Results

There really wasn't much to the EDA step, except that I just took a cursory look at the features.  In the pre-processing step, I used MinMaxScaler and LabelEncoder to scale the X features and to encode the y variable, and of course TrainTestSplit to get the training and test sets (33% default split).  Model development was also simple, as it only required initiating a sequential model, compiling it, and fitting the model to the categorical X training data with 60 epochs.  Quantifying the model simply involved computing loss and accuracy using evaluate().  In the Making Predictions step, I executed such predictions on scaled test dataset, and then converted the probabilities to class labels to view predictions alongside actuals.  The final step of evaluation entailed recomputing loss and accuracy manually by first computing how many false predictions were made, and finally taking a look at a confusion matrix using Scikit-Learn.

The last thing I did was simulate saving and loading the model on new data.  For that, I attempted to leverage ChatGPT on generating a new dataset.  Loading the model on this dataset was unsuccessful, as may be predictable, as the fake dataset failed to capture real patterns that were consistent with the original dataset.  Nevertheless, a template now exists for loading in another dataset in order to test the model's effectiveness on new data.

I am proud of my work here and am excited to continue to leverage the power of deep learning for other tasks!

### EXHIBITS

Exhibit A: Model Evaluation (Loss & Accuracy)

<img width="1189" alt="Exhibit A Model Evaluation" src="https://github.com/dsc55704973/tensorflow_voice_recognition/assets/66639071/85baa109-9f8a-48e9-ba95-300a34fd4463">

Exhibit B: Confusion Matrix

<img width="837" alt="Exhibit B Confusion Matrix" src="https://github.com/dsc55704973/tensorflow_voice_recognition/assets/66639071/c175366e-7d24-4612-a46f-a6a4ea6c634f">

Exhibit C: Predictions vs. Actuals (Encoded Values)

<img width="774" alt="Exhibit C Predictions vs  Actuals" src="https://github.com/dsc55704973/tensorflow_voice_recognition/assets/66639071/7ed9e258-c8f4-4d7a-9b59-b730758190e7">

### TECHNICAL

In order to run this code, you'll need to ensure you have Tensorflow, Scikit-Learn, and the other imported packages loaded on your machine.  

From your command line, run the following commands:

pip install tensorflow
pip install scikit-learn
pip install pandas
pip install numpy
pip install matplotlib

