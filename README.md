# Used Phone Price Prediction
### Overview
This repository contains code for a project focused on predicting used phone prices using machine learning techniques. The project involves data preprocessing, exploratory data analysis (EDA), and the implementation of various regression models.
### Introduction
The main goal in this project is to forecast the normalized costs of old phones by taking into account a variety of attributes, including internal memory, RAM, camera specs, and more. The following predictive models are employed: Random Forest, Lasso, Ridge, and Linear regression.


### Requirements
To run this project locally, you need to have Python installed. Additionally, install the required libraries using the following command:

```bash
pip install -r requirements.txt
```
### Application
To run the code step-by-step, launch the Jupyter Notebook  *recell_analysis.ipynb*. If your dataset is in a separate directory, be sure to correct the file path.

### Exploratory Data Analysis
The notebook includes thorough exploratory data analysis, visualizing the distribution of target variables, and exploring relationships between features and prices.

### Data Cleaning
This  section covers:
> - handling missing values
> - dealing with outliers 
> - encoding categorical variables.
> - creation of new features

### Model building
Regression models used in this project include:
> - Ridge regression
> - Lasso regression
> - Random Forest regression
> - Linear regression

Hyperparameter tuning, evaluation, and model training are all included in this section.

### Results
The root mean squared error (RMSE) for each model is provided after training each model, indicating how well the models perform in terms of used phone price prediction.


