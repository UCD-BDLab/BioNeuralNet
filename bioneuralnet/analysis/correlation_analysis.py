import pandas as pd


class Correlation:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def calculate_correlation(self) -> pd.DataFrame:
        """
        Calculates the correlation matrix for the input data.

        Returns:
            pd.DataFrame: The correlation matrix.
        """
        return self.data.corr()
