Breast Cancer Detection using Decision Trees

This project explores how a machine learning model can be used to classify breast cancer tumors as malignant or benign. I used a Decision Tree classifier from the Scikit-learn library and trained it on the Breast Cancer Wisconsin (Diagnostic) dataset. The dataset contains several features that are calculated from images of breast masses, such as radius, texture, and smoothness.

The goal of the project was to understand how decision trees work and how they can be applied to a real classification problem in healthcare.

Project Files

Decision Trees with Scikitlearn.ipynb
This Jupyter notebook contains the full implementation of the project. It includes data loading, preprocessing, model training, evaluation, and visualization.

README.md
This file provides a short description of the project and instructions on how to run it.

Main Steps in the Project

First, the breast cancer dataset was loaded from the sklearn.datasets module. After that, the data was explored and prepared for training. The dataset was split into training and testing sets so that the model could be evaluated properly.

A Decision Tree classifier from sklearn.tree was then trained using the training data. After training the model, its performance was evaluated on the test data. Different evaluation methods were used, including accuracy, a confusion matrix, and a classification report.

I also visualized the decision tree to better understand how the model makes decisions based on different features.

Tools and Libraries

The project was implemented using Python. The main libraries used are:

scikit-learn

pandas

numpy

matplotlib

seaborn

These libraries were used for data handling, machine learning, and visualization.

Results

The model achieved an accuracy of around 95% on the test dataset. The confusion matrix shows how many predictions were correct and how many were incorrect. By visualizing the decision tree, it is possible to see which features play an important role in the classification.

This helped me understand how the model splits the data based on different feature values to make predictions.

How to Run the Project

Clone the repository or download the notebook file.

Install the required Python libraries if they are not already installed:

pip install scikit-learn pandas numpy matplotlib seaborn

Open the notebook using Jupyter Notebook, Jupyter Lab, or VS Code.

Run the cells step by step to see the data analysis, model training, and evaluation.

Some improvements that could be explored in the future include tuning the hyperparameters of the decision tree, using cross-validation, and comparing the results with other machine learning models such as Random Forest or Support Vector Machines.
