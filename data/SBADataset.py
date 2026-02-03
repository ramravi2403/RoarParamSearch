import os
from typing import Tuple, List, Literal
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from Dataset import Dataset


class SBADataset(Dataset):
    """
    Unified SBA dataset processor with configurable feature sets and split strategies.

    Supports:
    - Minimal or full feature sets
    - Stratified or chronological splits
    - Configurable train/test/query ratios
    - Optional data leakage mode for experimentation
    """

    MINIMAL_FEATURES = [
        "Term", "NoEmp", "CreateJob", "RetainedJob",
        "Portion", "RealEstate", "RevLineCr"
    ]
    MINIMAL_CONT_COLS = ["Term", "NoEmp", "CreateJob", "RetainedJob", "Portion"]
    MINIMAL_BIN_COLS = ["RealEstate", "RevLineCr"]

    DROPPED_COLS = [
        "Selected", "State", "Name", "BalanceGross", "LowDoc",
        "BankState", "LoanNr_ChkDgt", "MIS_Status", "Default",
        "Bank", "City"
    ]

    TARGET = "NoDefault"

    def __init__(self,
                 fold: int = 0,
                 feature_mode: Literal["minimal", "full"] = "full",
                 split_strategy: Literal["stratified", "chronological"] = "chronological",
                 keep_approvalfy: bool = False,  # Changed default to False
                 scaler: StandardScaler | None = None):
        """
        Args:
            fold: Fold number for stratified splits
            feature_mode: "minimal" for 7 features, "full" for all features with OHE
            split_strategy: "stratified" or "chronological"
            keep_approvalfy: If True, keep ApprovalFY as a feature (default: False to prevent leakage)
                            Note: ApprovalFY is always used for chronological sorting when needed,
                            this only controls whether it appears in the final feature set.
            scaler: Optional pre-fitted scaler
        """
        super().__init__(fold=fold, scaler=scaler if scaler is not None else StandardScaler())
        self.feature_mode = feature_mode
        self.split_strategy = split_strategy
        self.keep_approvalfy = keep_approvalfy

    def get_data(
            self,
            file_name: str,
            save_dir: str = "data/sba",
            train_ratio: float = 0.6,
            test_ratio: float = 0.2,
            query_ratio: float = 0.2,
            introduce_leakage: bool = False,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Load, process, and split SBA data into train/test/query sets.

        Args:
            file_name: Path to CSV file
            save_dir: Directory to save .npy files
            train_ratio: Proportion for training (default 0.6)
            test_ratio: Proportion for testing (default 0.2)
            query_ratio: Proportion for query set (default 0.2)
            introduce_leakage: If True, scale before splitting (for experiments)

        Returns:
            X_train, y_train, X_test, y_test, X_query, y_query
        """
        if not np.isclose(train_ratio + test_ratio + query_ratio, 1.0):
            raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + test_ratio + query_ratio}")

        os.makedirs(save_dir, exist_ok=True)
        df = self.load_data(file_name)
        if self.feature_mode == "minimal":
            df, feature_names, cont_cols = self._process_minimal_features(df)
        else:
            df, feature_names, cont_cols = self._process_full_features(df)

        X_train, y_train, X_test, y_test, X_query, y_query = self._split_data(
            df, train_ratio, test_ratio, query_ratio
        )
        cont_cols = [col for col in cont_cols if col in X_train.columns]
        if introduce_leakage:
            X_combined = pd.concat([X_train, X_test, X_query], axis=0)
            if len(cont_cols) > 0:
                for col in cont_cols:
                    if col in X_combined.columns:
                        self.scaler.fit(X_combined[[col]])
                        X_train[col] = self.scaler.transform(X_train[[col]])
                        X_test[col] = self.scaler.transform(X_test[[col]])
                        X_query[col] = self.scaler.transform(X_query[[col]])
        else:
            if len(cont_cols) > 0:
                for col in cont_cols:
                    if col in X_train.columns:
                        scaler = StandardScaler()
                        X_train[col] = scaler.fit_transform(X_train[[col]])
                        X_test[col] = scaler.transform(X_test[[col]])
                        X_query[col] = scaler.transform(X_query[[col]])

        should_drop_approvalfy = not (self.feature_mode == "full" and self.keep_approvalfy)
        if should_drop_approvalfy and "ApprovalFY" in X_train.columns:
            X_train = X_train.drop(columns=["ApprovalFY"])
            X_test = X_test.drop(columns=["ApprovalFY"])
            X_query = X_query.drop(columns=["ApprovalFY"])

        feature_names = X_train.columns.tolist()

        self._save_datasets(
            save_dir, X_train, y_train, X_test, y_test,
            X_query, y_query, feature_names
        )

        return X_train, y_train, X_test, y_test, X_query, y_query

    def _process_minimal_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """Process dataset with minimal 7 features."""
        if self.TARGET not in df.columns:
            if "Default" not in df.columns:
                raise ValueError("Dataset must have either 'NoDefault' or 'Default' column.")
            df[self.TARGET] = 1 - df["Default"].astype(int)

        missing = [c for c in self.MINIMAL_FEATURES if c not in df.columns]
        if missing:
            if "Portion" in missing and {"SBA_Appv", "GrAppv"}.issubset(df.columns):
                df["Portion"] = (df["SBA_Appv"] / df["GrAppv"]).replace([np.inf, -np.inf], np.nan)
                missing = [c for c in self.MINIMAL_FEATURES if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

        cols_needed = self.MINIMAL_FEATURES + [self.TARGET]
        if "ApprovalFY" in df.columns:
            cols_needed.append("ApprovalFY")
        work = df[cols_needed].copy()

        work["RevLineCr"] = work["RevLineCr"].apply(self._normalize_revlinecr).astype(int)
        work["RealEstate"] = work["RealEstate"].fillna(0).astype(int).clip(0, 1)

        work["Portion"] = pd.to_numeric(work["Portion"], errors="coerce")
        work["Portion"] = work["Portion"].clip(lower=0.0, upper=1.0)

        work = work.dropna(subset=self.MINIMAL_FEATURES + [self.TARGET]).reset_index(drop=True)

        return work, self.MINIMAL_FEATURES.copy(), self.MINIMAL_CONT_COLS.copy()

    def _process_full_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Process dataset with full feature set and one-hot encoding.
        Mimics the ROAR paper's TemporalShift preprocessing.
        """
        if "Default" not in df.columns:
            raise ValueError("Dataset must have 'Default' column.")
        df[self.TARGET] = 1 - df["Default"].values
        df = df.fillna(-1)

        drop_cols = [col for col in self.DROPPED_COLS if col in df.columns]
        if "Default" in df.columns:
            drop_cols.append("Default")
        df = df.drop(columns=list(set(drop_cols)))
        work = df.drop(columns=[self.TARGET]).copy()
        if "ApprovalFY" in work.columns:
            work = work.drop(columns=["ApprovalFY"])
        cat_feat, num_feat = self._get_feat_types(work)
        df = pd.get_dummies(df, columns=cat_feat)
        if "ApprovalFY" in df.columns and "ApprovalFY" not in num_feat:
            if len(df["ApprovalFY"].unique()) > 2:
                num_feat = num_feat + ["ApprovalFY"]

        feature_names = [col for col in df.columns if col != self.TARGET]
        return df, feature_names, num_feat

    def _split_data(
            self,
            df: pd.DataFrame,
            train_ratio: float,
            test_ratio: float,
            query_ratio: float
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Split data into train/test/query sets using configured strategy."""

        if self.split_strategy == "chronological":
            return self._chronological_split(df, train_ratio, test_ratio, query_ratio)
        else:
            return self._stratified_three_way_split(df, train_ratio, test_ratio, query_ratio)

    def _chronological_split(
            self,
            df: pd.DataFrame,
            train_ratio: float,
            test_ratio: float,
            query_ratio: float
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Split data chronologically based on ApprovalFY.

        Note: Unlike the paper which creates two datasets (pre-2006 and full),
        we create a single chronological split of the full dataset.
        """
        if "ApprovalFY" not in df.columns:
            raise ValueError("DataFrame must contain 'ApprovalFY' for chronological split.")

        df_sorted = df.sort_values(by="ApprovalFY", ascending=True).reset_index(drop=True)

        n = len(df_sorted)
        train_end = int(train_ratio * n)
        test_end = train_end + int(test_ratio * n)

        X = df_sorted.drop(columns=[self.TARGET])
        y = df_sorted[self.TARGET]

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_test, y_test = X.iloc[train_end:test_end], y.iloc[train_end:test_end]
        X_query, y_query = X.iloc[test_end:], y.iloc[test_end:]

        return X_train, y_train, X_test, y_test, X_query, y_query

    def _stratified_three_way_split(
            self,
            df: pd.DataFrame,
            train_ratio: float,
            test_ratio: float,
            query_ratio: float
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Split data using stratified sampling."""
        from sklearn.model_selection import train_test_split

        X = df.drop(columns=[self.TARGET])
        y = df[self.TARGET].astype(int)

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(test_ratio + query_ratio),
            stratify=y,
            random_state=42 + self.fold
        )

        test_size_adjusted = test_ratio / (test_ratio + query_ratio)
        X_test, X_query, y_test, y_query = train_test_split(
            X_temp, y_temp,
            test_size=(1 - test_size_adjusted),
            stratify=y_temp,
            random_state=43 + self.fold
        )

        return X_train, y_train, X_test, y_test, X_query, y_query

    def _scale_dataframe(self, X: pd.DataFrame, cont_cols: List[str]) -> pd.DataFrame:
        """Apply fitted scaler to continuous columns."""
        if len(cont_cols) == 0:
            return X

        cols_to_scale = [col for col in cont_cols if col in X.columns]

        if len(cols_to_scale) == 0:
            return X

        X_scaled = X.copy()
        X_scaled[cols_to_scale] = self.scaler.transform(X[cols_to_scale])
        return X_scaled

    def _save_datasets(
            self,
            save_dir: str,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_test: pd.DataFrame,
            y_test: pd.Series,
            X_query: pd.DataFrame,
            y_query: pd.Series,
            feature_names: List[str]
    ):
        """Save all datasets to .npy files."""
        train_arr = self._pack_xy(y_train, X_train, feature_names)
        test_arr = self._pack_xy(y_test, X_test, feature_names)
        query_arr = self._pack_xy(y_query, X_query, feature_names)

        np.save(os.path.join(save_dir, "sba_train.npy"), train_arr, allow_pickle=False)
        np.save(os.path.join(save_dir, "sba_test.npy"), test_arr, allow_pickle=False)
        np.save(os.path.join(save_dir, "sba_query.npy"), query_arr, allow_pickle=False)
        np.save(
            os.path.join(save_dir, "sba_feature_names.npy"),
            np.array(feature_names, dtype=object),
            allow_pickle=True
        )

        print(f"✅ Saved datasets to {save_dir}/")
        print(f"   Train: {train_arr.shape}, Test: {test_arr.shape}, Query: {query_arr.shape}")

    @staticmethod
    def _pack_xy(y: pd.Series, X: pd.DataFrame, feature_order: List[str]) -> np.ndarray:
        """Pack y and X into single array: [y, feature1, feature2, ...]"""
        X = X[feature_order].to_numpy(dtype=np.float32)
        y = y.to_numpy(dtype=np.float32).reshape(-1, 1)
        return np.concatenate([y, X], axis=1)

    @staticmethod
    def _normalize_revlinecr(x) -> int:
        """Normalize RevLineCr to binary 0/1."""
        if pd.isna(x):
            return 0
        s = str(x).strip().upper()
        if s in {"Y", "YES", "T", "TRUE", "1"}:
            return 1
        return 0

    @staticmethod
    def _get_feat_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Identify categorical and numerical features."""
        cat_feat = []
        num_feat = []
        for key in df.columns:
            if df[key].dtype == object:
                cat_feat.append(key)
            elif len(df[key].unique()) > 2:
                num_feat.append(key)
        return cat_feat, num_feat