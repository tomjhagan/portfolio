# import relevant libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def box_plots(df, n_row, n_col):
    """
    Displays a boxplot for each feature in a DataFrame, split by cluster label

    Parameters:
    - df (pd.DataFrame): The dataframe for analysis containing cluster labels
    - n_row (int): Desired number of rows of subplots
    - n_col (int): Desired number of columns of subplots
    """
    # Set subplot number
    plot_num=1

    plt.figure(figsize=(15, 20))

    # Exclude 'Cluster' from columns to plot
    columns_to_plot = [col for col in df.columns if col != 'Cluster']

    # Set color palette for clusters
    unique_clusters = sorted(df['Cluster'].unique())
    palette = sns.color_palette("Set2", len(unique_clusters))

    # Plot a box plot for each variable
    for column in columns_to_plot:
        plt.subplot(n_row, n_col, plot_num)
        sns.boxplot(data=df, x='Cluster', y=column, palette=palette)
        plt.title(column)

        # Change the subplot number for next plot
        plot_num += 1

    # Show the plot
    plt.show()

def plot_pca(X, y):
    """
    Transforms high-dimensional data and visualises
    the different clusters after PCA dimensionality reduction.

    Parameters:
    - X: Scaled DataFrame with shape (n_samples, n_features)
    - y: Cluster labels from K-means
    """
    # Apply PCA to reduce the data set to two components
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(X)

    # Transform into a DataFrame
    df_pca = pd.DataFrame({'PCA 1': df_pca[:, 0], 'PCA 2': df_pca[:, 1]})

    # Label the data based on the different clusters from K-means
    df_pca["Cluster"] = y

    # Visualise the output
    sns.scatterplot(data=df_pca, x="PCA 1", y="PCA 2", hue="Cluster", palette="tab10")
    plt.show()

def plot_tsne(X, y):
    """
    Transforms high-dimensional data and visualises
    the different clusters after t-SNE dimensionality reduction.

    Parameters:
    - X: Scaled DataFrame with shape (n_samples, n_features)
    - y: Cluster labels from K-means
    """
    # Transform the data with t-SNE.
    TSNE_model = TSNE(n_components=2, perplexity=30.0)
    TSNE_transformed_data = TSNE_model.fit_transform(X)

    # Transform into a DataFrame
    df_tsne = pd.DataFrame({'Component 1': TSNE_transformed_data[:, 0], 'Component 2': TSNE_transformed_data[:, 1]})

    # Label the data based on the different clusters from K-means
    df_tsne["Cluster"] = y
    # Visualise the output.
    sns.scatterplot(data=df_tsne, x="Component 1", y="Component 2", hue="Cluster", palette="tab10")
    plt.show()
