import pandas as pd
import matplotlib.pyplot as plt
from ..utils.logger import get_logger

class CheckVariance:
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize with the dataframe whose feature variances you want to analyze.
        
        :param dataframe: A pandas DataFrame containing the data.
        """
        self.df = dataframe
        self.logger = get_logger(__name__)
        self.logger.info(f"CheckVariance initialized with dataframe of shape {self.df.shape}")

    def plot_variance_distribution(self, bins: int = 50):
        """
        Compute the variance for each feature (column) in the DataFrame and plot
        a histogram of these variances.

        :param bins: Number of bins for the histogram.
        :return: A matplotlib Figure object containing the plot.
        """
        variances = self.df.var()
        self.logger.info("Computed variances for each feature.")

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(variances, bins=bins, edgecolor='black')
        ax.set_title("Distribution of Feature Variances")
        ax.set_xlabel("Variance")
        ax.set_ylabel("Frequency")
        
        self.logger.info("Variance distribution plot generated.")
        return fig

    def plot_variance_by_feature(self):
        """
        Plot the variance for each feature against its index or name.
        
        :return: A matplotlib Figure object containing the plot.
        """
        variances = self.df.var()
        self.logger.info("Computed variances for each feature for index plot.")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(variances.index, variances.values, 'o', markersize=4)
        ax.set_title("Variance per Feature")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Variance")
        ax.tick_params(axis='x', rotation=90)
        
        self.logger.info("Variance vs. feature index plot generated.")
        return fig

    def remove_near_zero_variance(self,variance_threshold=1e-6):
        self.logger.info()
        variances = self.df.var()
        return self.df.loc[:, variances > variance_threshold]

    def remove_high_zero_fraction(self, zero_frac_threshold=0.95):
        zero_fraction = (self.df == 0).sum(axis=0) / self.df.shape[0]
        return self.df.loc[:, zero_fraction < zero_frac_threshold]

def network_remove_low_variance(network, threshold=1e-6):
    """
    Removes rows and columns from a symmetric adjacency matrix 
    where the variance is below a specified threshold.
    
    Parameters:
        network (pd.DataFrame): The input symmetric adjacency matrix.
        threshold (float): Variance threshold below which rows/columns are removed.
    
    Returns:
        pd.DataFrame: The filtered adjacency matrix.
    """
    variances = network.var(axis=0)
    valid_indices = variances[variances > threshold].index
    return network.loc[valid_indices, valid_indices]

def network_remove_high_zero_fraction(network, threshold=0.95):
    """
    Removes rows and columns from a symmetric adjacency matrix 
    where the fraction of zero entries is higher than a specified threshold.
    
    Parameters:
        network (pd.DataFrame): The input symmetric adjacency matrix.
        threshold (float): Zero-fraction threshold above which rows/columns are removed.
    
    Returns:
        pd.DataFrame: The filtered adjacency matrix.
    """
    zero_fraction = (network == 0).sum(axis=0) / network.shape[0]
    valid_indices = zero_fraction[zero_fraction < threshold].index
    return network.loc[valid_indices, valid_indices]


