# Loan Approval Prediction Training Using Variuos Machine Learning Models

## Overview

This repository contains a Jupyter Notebook and an web-app with Python for loan approval prediction using machine learning techniques such as data preprocessing, feature scaling, PCA, and SMOTE.

## Requirements

To run this project smoothly, you need to install the following Python libraries and extensions:

### Required Libraries

You can install all required dependencies using the following command:

```bash
pip install streamlit numpy pandas scikit-learn imbalanced-learn matplotlib seaborn tensorflow flask joblib keras-tuner jupyter ipykernel shap graphviz xgboost
```

Alternatively, you can install them individually:

```bash
pip install numpy pandas
pip install scikit-learn
pip install imbalanced-learn
pip install matplotlib seaborn
.........
```

### VS Code Setup

Ensure you have **Visual Studio Code (VS Code)** installed along with the **Python extension (Pylance)**:

1. Download and install [VS Code](https://code.visualstudio.com/).
2. Install the Python and Pylance extension from the Extensions Marketplace.
3. Ensure you have Python installed:
   ```bash
   python --version
   ```
4. Open the project folder in VS Code.
5. Select the Python interpreter (Ctrl + Shift + P → "Python: Select Interpreter").

## File Descriptions

- `loan-approval-test.py`: The main Python script containing data processing, feature engineering, and model training.
- `processed_dataset.csv`: The dataset after feature encoding, scaling, and PCA transformation.
- `resampled_train_dataset.csv`: The training dataset after applying SMOTE.
- `loan_approval_test.ipynb`: The Jupyter Notebook containing the code for data preprocessing and training
- `app_test.py`: The Python script for the web application using Streamlit.
- `pca.pkl`: The saved PCA model that was used in the training for the web application
- `scaler.pkl`: The saved Feature Scaling model that was used in the training for the web application

## Running the Training Script

1. Open VS Code.
2. Open the project folder.
3. Selecting Run All at the actions bar on top of the file

## Running the Web-App

1. Open the terminal in VS Code (Ctrl + `).
2. Navigate to the project directory using cd
3. Run the following command to start the web app:
   ```bash
   streamlit run app_test.py
   ```

## Notes

- Make sure you have the dataset in the correct directory before running the script.
- The dataset file path might need adjustment depending on your system.
- You can delete untitled_project folder to training ANN model completely from the start again

## Contact

If you have any issues or questions, feel free to reach out with us on [Github](https://github.com/Shwooshie/LoanPredictionAI/tree/main)!
