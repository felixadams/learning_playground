import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from imblearn.over_sampling import SMOTE

# Function to load the dataset
def load_data(file_path):
    """Load a CSV file into a DataFrame."""
    df = pd.read_csv(file_path)
    return df

# Function to manually set binary or multi-class labels
def set_labels(y):
    """Convert categorical labels to binary or multi-class, return labels and binarizer."""
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)  # Convert labels to integers
    class_names = le.classes_  # Get class names for plotting
    return y_encoded, class_names

# Function to plot ROC curve for binary classification
def plot_roc_curve(y_test, y_pred_proba):
    """Plot the ROC curve for the positive class (binary)."""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    st.pyplot(plt.gcf())
    plt.clf()

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot the confusion matrix with class names."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label') 
    plt.xticks(rotation=90)
    st.pyplot(plt.gcf())
    plt.clf()

# Function to plot box plot of AUC scores
def plot_auc_boxplot(auc_scores):
    """Plot a box plot of AUC scores from cross-validation."""
    plt.figure()
    sns.boxplot(data=auc_scores)
    plt.title('AUC Scores from Cross-Validation')
    plt.ylabel('AUC')
    st.pyplot(plt.gcf())
    plt.clf()


def plot_pca(X_train_scaled, y_train):
    # Plot the PCA plot coloring the different groups
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train_scaled)
    X_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    X_pca['type'] = y_train
    plt.figure()
    explained_variance_1 = round(pca.explained_variance_ratio_[0], 2)
    explained_variance_2 = round(pca.explained_variance_ratio_[1], 2)
    sns.scatterplot(data=X_pca, x='PC1', y='PC2', hue='type', palette='viridis')
    plt.xlabel(f'PC1 ({explained_variance_1*100}%)')
    plt.ylabel(f'PC2 ({explained_variance_2*100}%)')
    plt.title('PCA Plot of Training Data')
    st.pyplot(plt.gcf())
    plt.clf()

# Streamlit app - Main function
def main():
    """Main function to run the Streamlit app."""
    st.title("Machine Learning Model Evaluation with Cross-Validation and PCA")

    # Introduction
    st.write("""This application allows you to evaluate various machine learning models for classification tasks using your dataset.""")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load the data
        df = load_data(uploaded_file)
        st.write("Data Preview:", df.head())  # Show a preview of the data

        # Ensure that the 'type' column exists
        if 'type' not in df.columns:
            st.error("The dataset must contain a 'type' column for the output variable.")
            return

        # Select unique target classes
        target_classes = st.multiselect("Select target classes", options=df['type'].unique())
        if len(target_classes) < 2:
            st.error("Please select at least two target classes.")
            return

        # Filter the dataset based on selected target classes
        df_filtered = df[df['type'].isin(target_classes)]

        # Select train-test split ratio
        test_size = st.slider("Test Set Ratio", min_value=0.1, max_value=0.99, value=0.2, step=0.05)

        # Dropdown to select the model
        model_choice = st.selectbox(
            "Choose a machine learning model:",
            ("Logistic Regression", "Random Forest", "SVM Classifier", "Multi-Layer Perceptron")
        )

        # Number of cross-validation folds
        num_folds = st.slider("Number of Cross-Validation Folds", min_value=3, max_value=10, value=5, step=1)

        # Button to train model
        if st.button("Train Model"):
            # Separate features and target
            X = df_filtered.drop(columns=['type', 'samples'])  # Features
            y = df_filtered['type']  # Target

            smote = SMOTE()
            X, y = smote.fit_resample(X, y)


            # Encode the labels
            y_encoded, class_names = set_labels(y)

            # Split the dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded)

            # Standardize the data
            scaler = StandardScaler()

            # Apply the scaler to the numerical columns
            X_train_numeric = X_train.select_dtypes(include=[np.float64, np.float32])  # Select only float columns
            X_test_numeric = X_test.select_dtypes(include=[np.float64, np.float32])

            X_train_scaled = scaler.fit_transform(X_train_numeric)
            X_test_scaled = scaler.transform(X_test_numeric)

            # Apply PCA
            pca = PCA(n_components=0.95)
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_test_pca = pca.transform(X_test_scaled)

            st.write(f"PCA reduced the number of features to {X_train_pca.shape[1]}.")

            # Select and train the model
            if model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=500, class_weight='balanced')
            elif model_choice == "Random Forest":
                model = RandomForestClassifier(class_weight='balanced')
            elif model_choice == "SVM Classifier":
                model = SVC(probability=True, class_weight='balanced')
            elif model_choice == "Multi-Layer Perceptron":
                hidden_layer_1 = st.slider("Number of Neurons in Hidden Layer 1", min_value=10, max_value=100, value=50, step=10)
                hidden_layer_2 = st.slider("Number of Neurons in Hidden Layer 2", min_value=10, max_value=100, value=50, step=10)
                model = MLPClassifier(hidden_layer_sizes=(hidden_layer_1, hidden_layer_2), max_iter=500)


            # Choose scoring method based on number of target classes
            if len(target_classes) == 2:
                # Binary classification
                scoring = 'roc_auc'
            else:
                # Multiclass classification
                scoring = 'roc_auc_ovr'  # One-vs-rest ROC AUC for multiclass


            # Cross-validation with AUC calculation
            skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
            auc_scores = cross_val_score(model, X_train_pca, y_train, cv=skf, scoring=scoring)

            # Train the model on the full training set and predict on the test set
            model.fit(X_train_pca, y_train)
            y_pred_proba = model.predict_proba(X_test_pca)  # Get probabilities for all classes

            # ROC curve for binary classification
            if len(target_classes) == 2:
                plot_roc_curve(y_test, y_pred_proba)

            # Confusion matrix
            plot_confusion_matrix(y_test, model.predict(X_test_pca), class_names)

            # Plot AUC box plot
            st.subheader("AUC Scores from Cross-Validation")
            plot_auc_boxplot(auc_scores)

            plot_pca(X_train_scaled, y_train)

            # Display a table with test set results
            results_df = pd.DataFrame({
                'True Label': y_test,
                'Predicted Probability (Class 1)': y_pred_proba[:, 1] if len(target_classes) == 2 else None,
                'Predicted Label': model.predict(X_test_pca)
            })
            st.subheader("Test Set Predictions")
            st.write(results_df)

if __name__ == "__main__":
    main()
