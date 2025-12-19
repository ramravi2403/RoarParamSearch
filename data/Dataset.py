import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Dataset(object):
    """
    Base dataset with:
      - CSV loading
      - stratified train/test split
      - single StandardScaler fit on TRAIN continuous block
    """
    def __init__(self, fold: int = 0, scaler: StandardScaler | None = None):
        self.fold = int(fold)
        self.scaler = scaler if scaler is not None else StandardScaler()

    def load_data(self, fname: str, sep: str = ",") -> pd.DataFrame:
        df = pd.read_csv(fname, sep=sep)
        df = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
        return df

    def stratified_split(
        self,
        df: pd.DataFrame,
        target_col: str,
        test_size: float = 0.2,
        base_random_state: int = 42,
    ):
        rs = base_random_state + self.fold  # different seed per "fold"
        X = df.drop(columns=[target_col])
        y = df[target_col].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=rs
        )
        return X_train, y_train, X_test, y_test

    def scale_continuous(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        cont_cols: list[str],
    ):
        if len(cont_cols) == 0:
            return X_train, X_test, self.scaler

        scaler = self.scaler
        scaler.fit(X_train[cont_cols].values)

        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[cont_cols] = scaler.transform(X_train[cont_cols].values)
        X_test_scaled[cont_cols] = scaler.transform(X_test[cont_cols].values)
        self.scaler = scaler
        return X_train_scaled, X_test_scaled, scaler

    def get_scaler(self) -> StandardScaler:
        return self.scaler
