# Cancer Dataset Classification Web App
This web app provides an interface for analyzing and classifying cancer gene expression data using various machine learning (ML) models. The app is designed to help users compare model performance across different cancer types, identify influential genes, and perform feature selection to enhance classification accuracy. The focus is on both binary classification (cancer vs. non-cancer) and multiclass classification (different cancer subtypes).
## Features
### File Upload: 
Upload a CSV dataset for model training and testing.

### Data Preprocessing:
Includes SMOTE for data balancing, PCA for dimensionality reduction, and data standardization.

### Model Selection:
Choose from multiple classification models:

Logistic Regression

Random Forest

Support Vector Machine (SVM)

Multi-Layer Perceptron (MLP)

### Cross-Validation:
Perform k-fold cross-validation with AUC scoring.

### Visualizations:
ROC Curve for binary classification.

Confusion Matrix for all classes.

AUC Score Box Plot.

PCA Plot for data visualization in 2D space.

### Prerequisites

Python 3.8 or above

pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit, and imblearn


## Future Expansion
### Potential enhancements for the app include:

### Additional Feature Selection Techniques: 
Integrate feature selection techniques, such as L1 regularization (lasso) or more advanced deep learning methods, for identifying relevant genes.
### Exploratory Data Analysis (EDA):
Add tools for visualizing distributions and correlations within the dataset to provide additional insights before model training.
### Hyperparameter Optimization:
Incorporate automated hyperparameter tuning (e.g., Grid Search or Bayesian Optimization) to find the best model configurations for each dataset.
### Cross-validation:
Allow k-fold cross-validation to provide a more robust assessment of model performance across different data splits.
