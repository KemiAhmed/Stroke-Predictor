
# Explainable Machine Learning Models for Stroke Prediction with Class Imbalance

## Project Overview
This project explores the use of explainable machine learning models for stroke prediction, specifically addressing the challenge of class imbalance. The goal is to predict the likelihood of a stroke occurring in patients using clinical data while ensuring that the models remain interpretable. This is crucial for healthcare applications where the decisions made by the models should be understandable to medical professionals. 

To handle the class imbalance, a hybrid sampling approach combining **ROSE (Random OverSampling Examples)** and **OVUN.SAMPLE** is employed. The machine learning models used in this project include:
- **Logistic Regression**
- **Random Forest**
- **Decision Trees**
- **XGBoost**
- **Support Vector Machines (SVM)**

Finally, the project includes a **Stroke Prediction Dashboard** built using **Shiny App** to allow users to interact with the model and visualize predictions in an easy-to-use interface.

## Table of Contents
- [Background](#background)
- [Data](#data)
- [Methods](#methods)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Shiny App Dashboard](#shiny-app-dashboard)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Background
Stroke is a leading cause of death and long-term disability globally. Early detection of stroke risk can significantly improve patient outcomes. However, predicting stroke is a complex task, especially with imbalanced datasets where the number of non-stroke cases vastly outweighs stroke cases.

Machine learning models are often used for such predictions, but they can struggle with class imbalance. This project addresses this by employing hybrid sampling techniques and several machine learning models. Additionally, model explainability is emphasized through SHAP (SHapley Additive exPlanations), which is used both for global feature importance and local explanation of individual predictions.

## Data
This project uses publicly available stroke prediction datasets, such as the **Stroke Prediction Dataset** from Kaggle. The dataset includes features like:

- Age
- Gender
- Hypertension
- Heart disease
- Marital status
- Work type
- Residence type
- Average glucose level
- Body mass index (BMI)
- Smoking status

The target variable indicates whether a patient has had a stroke (1) or not (0). To address the class imbalance, a hybrid sampling technique involving **ROSE (Random OverSampling Examples)** and **OVUN.SAMPLE** is applied.

## Methods
The project employs several machine learning algorithms for stroke prediction:

1. **Logistic Regression**  
   A baseline model that is interpretable and useful for linear classification.

2. **Random Forest Classifier**  
   An ensemble learning method that builds multiple decision trees to improve predictive performance, particularly in imbalanced datasets.

3. **Decision Trees**  
   A tree-based model that is intuitive and easy to interpret, but may overfit if not properly tuned.

4. **XGBoost**  
   A highly efficient gradient boosting algorithm known for its predictive power, especially in dealing with imbalanced data.

5. **Support Vector Machines (SVM)**  
   A powerful classifier that is effective for high-dimensional spaces and imbalanced data.

### Addressing Class Imbalance:
To handle class imbalance, the following hybrid sampling methods are used:
- **ROSE (Random OverSampling Examples)**: Generates synthetic examples for the minority class to balance the dataset.
- **OVUN.SAMPLE**: A combination of undersampling of the majority class and oversampling of the minority class.

### Explainability with SHAP:
SHAP (SHapley Additive exPlanations) is used for both global and local interpretability:
- **Global Explanations**: SHAP is used to understand the overall feature importance across the entire dataset, helping us determine which features most influence the model's decision-making.
- **Local Explanations**: SHAP is also applied to individual predictions, providing a breakdown of how each feature contributed to a specific prediction (e.g., whether a patient is at high risk for stroke).

## Results
The following performance metrics are used to evaluate the models:
- **Accuracy**: The overall classification performance of the model.
- **Precision, Recall, and F1-Score**: Metrics for evaluating model performance on the minority class (stroke cases).
- **AUC-ROC (Area Under the Receiver Operating Characteristic Curve)**: Measures the model’s ability to distinguish between stroke and non-stroke cases.
- **SHAP Explanations**: Both global and local SHAP values are computed to understand model predictions and feature importance.

## Installation

### Prerequisites
Ensure you have Python 3.6+ and R installed for running the Shiny app, along with the following libraries:
- **Python Libraries**:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `xgboost`
  - `imbalanced-learn`
  - `shap`
  - `lime`
  - `matplotlib`
  - `seaborn`
  
- **R Libraries** (for the Shiny App):
  - `shiny`
  - `shinydashboard`
  - `plotly`
  - `shapr`
  
### Clone the Repository
```bash
git clone https://github.com/yourusername/stroke-prediction-explainable-models.git
cd stroke-prediction-explainable-models
```

### Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Install R Dependencies for Shiny App
```r
install.packages("shiny")
install.packages("shinydashboard")
install.packages("plotly")
install.packages("shapr")
```

## Usage

1. **Data Preprocessing**:
   Data preprocessing, including handling missing values, encoding categorical variables, and applying the hybrid sampling method (ROSE + OVUN.SAMPLE), is implemented in `preprocessing.py`.

2. **Model Training**:
   The script for training the models is `train_model.py`. You can specify the model to train and apply the hybrid sampling technique as follows:
   ```bash
   python train_model.py --model random_forest --sampling hybrid
   ```

3. **Model Evaluation**:
   Evaluate the performance of the trained models using `evaluate_model.py`.

4. **Model Explanation with SHAP**:
   Generate global and local explanations for the models' predictions using SHAP in `explain_model.py`.

   Example command to explain a specific prediction:
   ```bash
   python explain_model.py --model random_forest --instance 42 --method shap
   ```

## Shiny App Dashboard
The final component of this project is a **Stroke Prediction Dashboard** built using **Shiny App**. The dashboard allows users to interact with the model, input patient data, and receive real-time stroke risk predictions. The model’s predictions are accompanied by an explanation of which features influenced the decision, using SHAP for both global and local explanations.

To launch the Shiny App:

1. Open the R console and navigate to the project folder.
2. Run the Shiny app:
   ```r
   shiny::runApp("path/to/your/shiny_app")
   ```

The app provides a user-friendly interface to explore the stroke prediction model and visualizations.

## Contributing
Contributions are welcome! If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Clone your fork to your local machine.
3. Create a new branch (`git checkout -b feature-name`).
4. Make your changes and commit them (`git commit -am 'Add feature'`).
5. Push to your branch (`git push origin feature-name`).
6. Create a pull request explaining your changes.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- **Kaggle** for providing the stroke prediction dataset.
- **Scikit-learn**, **XGBoost**, and other machine learning libraries.
- **ROSE** and **OVUN.SAMPLE** for class imbalance handling techniques.
- The authors of **SHAP** for providing a powerful tool for model explainability.
- **R Shiny** for building the interactive Stroke Prediction Dashboard.
