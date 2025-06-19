import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

def box_plots(df, n_row, n_col):
    """
    Displays a boxplot for each feature in a DataFrame

    Parameters:
    - df (pd.DataFrame): The dataframe for analysis
    - n_row (int): Desired number of rows of subplots
    - n_col (int): Desired number of columns of subplots
    """
    # Set subplot number
    plot_num=1

    plt.figure(figsize=(15, 7))

    # Plot a box plot for each variable
    for column in df:
        plt.subplot(n_row, n_col, plot_num)
        sns.boxplot(df[column])
        plt.title(column)

        # Change the subplot number for next plot
        plot_num += 1

    # Show the plot
    plt.show()


def hist_plots(df, n_row, n_col):
    """
    Displays a histogram for each feature in a DataFrame

    Parameters:
    - df (pd.DataFrame): The dataframe for analysis
    - n_row (int): Desired number of rows of subplots
    - n_col (int): Desired number of columns of subplots
    """
    # Set subplot number
    plot_num = 1

    plt.figure(figsize=(15, 12))

    # Plot a hist plot for each variable
    for column in df:
        plt.subplot(n_row, n_col, plot_num)
        sns.histplot(df[column])
        plt.title(column)

        # Change the subplot number for next plot
        plot_num += 1

    # View the plot.
    plt.show()


def iqr(df):
    """
    Calculates and displays the interquartile range for values of a specified column/feature and creates a binary column displaying which data points lie outside that range.
    Args:
        df (pd.DataFrame): The DataFrame that is being analysed.
    """

    for column in df:
        # Sort the dataframe by the values of the feature being analysed
        sorted = df[column].sort_values()

        # Calculate the quantiles.
        q1 = sorted.quantile(0.25)
        q3 = sorted.quantile(0.75)

        # View the output.
        print(f"The value of Q1 for {column} is: {q1:.2f}")
        print(f"The value of Q3 for {column} is: {q3:.2f}")

        # Compute the IQR.
        iqr = q3 - q1
        print(f"The value of IQR is: {iqr:.2f}")

        # Calculate the upper and lower limits.
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        print(f"Upper limit for {column}: {upper:.2f}")
        print(f"Lower limit for {column}: {lower:.2f}\n")

        # Populate a binary column for each feature indicating whether the data falls outside of the IQR
        df[f"{column} outlier"] = df[column].apply(lambda x: int(x < lower) | int(x > upper))


def plot_ocsvm_with_pca(X, nu=0.02, kernel="rbf", gamma=0.5):
    """
    Trains a One-Class SVM on high-dimensional data and visualizes
    the decision boundary and outliers after PCA dimensionality reduction.

    Parameters:
    - X: numpy array or DataFrame with shape (n_samples, n_features)
    - nu: An upper bound on the fraction of training errors (outliers)
    - kernel: Kernel type for SVM
    - gamma: Kernel coefficient
    """

    # Fit One-Class SVM
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    model.fit(X)

    # Predict labels (-1: outlier, 1: inlier)
    preds = model.predict(X)

    # Reduce to 2D with PCA for plotting
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Create grid for decision boundary
    xx, yy = np.meshgrid(
        np.linspace(X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1, 500),
        np.linspace(X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1, 500)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Project grid back to original feature space (approximate inverse PCA)
    grid_original = pca.inverse_transform(grid)

    # Compute decision function
    Z = model.decision_function(grid_original)
    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 10), cmap=plt.cm.Blues_r)
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')

    # Plot inliers and outliers
    plt.scatter(X_2d[preds == 1, 0], X_2d[preds == 1, 1], c='white', s=20, edgecolors='k', label='Inliers')
    plt.scatter(X_2d[preds == -1, 0], X_2d[preds == -1, 1], c='red', s=20, edgecolors='k', label='Outliers')

    plt.title("One-Class SVM with PCA-reduced Data")
    plt.legend()
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.show()


def plot_isolation_forest_with_pca(X, contamination=0.025, n_estimators=100, random_state=42):
    """
    Trains an Isolation Forest on high-dimensional data and visualizes
    the decision boundary and outliers after PCA dimensionality reduction.

    Parameters:
    - X: numpy array or DataFrame with shape (n_samples, n_features)
    - contamination: Proportion of outliers in the data
    - n_estimators: Number of trees in the forest
    - random_state: For reproducibility
    """

    # Fit Isolation Forest
    model = IsolationForest(contamination=contamination,
                            n_estimators=n_estimators,
                            random_state=random_state)
    model.fit(X)

    # Predict labels (-1: outlier, 1: inlier)
    preds = model.predict(X)

    # Reduce to 2D with PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Create mesh grid in 2D PCA space
    xx, yy = np.meshgrid(
        np.linspace(X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1, 500),
        np.linspace(X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1, 500)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Project grid back to original space
    grid_original = pca.inverse_transform(grid)

    # Compute anomaly score (higher means more abnormal)
    Z = model.decision_function(grid_original)
    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(10, 6))

    # Plot inliers and outliers
    plt.scatter(X_2d[preds == 1, 0], X_2d[preds == 1, 1], c='white', s=20, edgecolors='k', label='Inliers')
    plt.scatter(X_2d[preds == -1, 0], X_2d[preds == -1, 1], c='red', s=20, edgecolors='k', label='Outliers')

    plt.title("Isolation Forest with PCA-reduced Data")
    plt.legend()
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.show()
