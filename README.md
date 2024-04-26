# Problem Statement

The objective of this project is to develop a predictive model that accurately classifies breast tumor diagnoses as either benign or malignant based on various features extracted from diagnostic images and patient data. The project aims to identify the best machine learning model and its optimal parameters to achieve high prediction accuracy.

## Tech Stack

- Python
- Jupyter Notebook
- pandas
- NumPy
- scikit-learn
- matplotlib

## Data Source

- [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

## Content

The dataset consists of various features derived from diagnostic tests, including radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.  They describe the characteristics of the cell nuclei present in the image. A few of the images can be found at this [link](http://www.cs.wisc.edu/~street/images/)

# Approach

1. Data Collection and Preprocessing:
- Obtain a dataset containing diagnostic features such as tumor size, shape, texture, and patient characteristics.
- Preprocess the data by handling missing values, encoding categorical variables, and scaling numerical features as necessary.
2. Model Selection and Optimization:
- Explore a variety of machine learning models suitable for binary classification tasks, such as Logistic Regression, Random Forest, Support Vector Machines, etc.
- Utilize techniques like cross-validation and grid search to identify the best-performing model and tune its hyperparameters for optimal performance.
3. Evaluation Metrics:
- Evaluate model performance using appropriate evaluation metrics such as accuracy, precision, recall, and F1-score.
- Select the model with the highest overall performance as the best model for the task.
4. Deployment with Streamlit:
- Develop a user-friendly web application using Streamlit framework to showcase the predictive capabilities of the best model.
- Integrate the trained model into the web app to provide real-time predictions and probabilities of tumor diagnoses being benign or malignant.
- Display additional insights and visualizations to aid interpretation of the model predictions.

## Screenshots

![App Screenshot](https://github.com/VishShaji/Breast-Cancer-Detections-Machine-Learning-Classification/blob/main/Assets/ModelEvaluation.png)

## Result

Achieved the highest accuracy of 97.37% in classifying Malignant and Benign Tumours with Logistic Regression.


## License

[MIT License](https://choosealicense.com/licenses/mit/)

Copyright (c) 2024 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Learn more about [MIT License](https://choosealicense.com/licenses/mit/).
