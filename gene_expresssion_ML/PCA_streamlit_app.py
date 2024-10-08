# web app that takes gene expression data and returns a PCA plot. 


import pandas as pd
from sklearn.decomposition import PCA
import streamlit as st
import matplotlib.pyplot as plt


# Function to load and preprocess the dataset
def load_data(file_path):
    """Load a CSV file into a DataFrame."""
    df = pd.read_csv(file_path)
    return df

# Function to perform PCA and plot the results
def plot_pca(df):
    """Perform PCA on the DataFrame and plot the results."""
    # Ensure that the 'type' column exists
    if 'type' not in df.columns:
        st.error("The dataset must contain a 'type' column for grouping.")
        return

    # Separate features and target
    X = df.drop(columns=['type'])  # Features
    y = df['type']  # Target

    # Perform PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Plot PCA
    plt.figure(figsize=(10, 8))
    explained_variance = pca.explained_variance_ratio_

    # Use a scatter plot for PCA results
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y.factorize()[0], cmap='Set1', edgecolors='k', s=50)

    # Create a custom legend
    unique_types = y.unique()
    color_map = plt.cm.get_cmap('Set1', len(unique_types))  # Colormap

    # Create legend items
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=label, 
                     markerfacecolor=color_map(i), markersize=10)
        for i, label in enumerate(unique_types)
    ]

    plt.legend(handles=handles, title="Type")
    plt.title(f'PCA (PC1: {explained_variance[0]*100:.2f}% var, PC2: {explained_variance[1]*100:.2f}% var)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    # Display the plot in Streamlit
    st.pyplot(plt.gcf())
    plt.clf()  # Clear the plot after displaying in Streamlit

# Streamlit app
def main():
    """Main function to run the Streamlit app."""
    st.title("PCA Visualization of Gene Expression Data")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load the data
        df = load_data(uploaded_file)
        st.write("Data Preview:", df.head())  # Show a preview of the data

        # PCA button
        if st.button("Perform PCA"):
            plot_pca(df)

if __name__ == "__main__":
    main()
