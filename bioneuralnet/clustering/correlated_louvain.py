import pandas as pd


class Louvain:
    """
    Correlated Louvain:

    Attributes:
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def run(self) -> pd.DataFrame:
        """

        Returns:
            pd.DataFrame:
        """
        return self.data
