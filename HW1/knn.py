from collections import Counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, k: int):
        self.k = k
        self.features = pd.DataFrame([])
        self.labels = pd.Series([])
        self.index = pd.Index([])
        self.target = ""
        self.columns = pd.Index([])
        self.num_cols = pd.Index([])
        self.cat_cols = pd.Index([])

    def train(self, X: pd.DataFrame, y: pd.Series):
        # Sanity check
        assert all(X.index == y.index), "Indices mismatch"
        # Drop rows with missing data
        Xy = pd.concat([X, y], axis=1).dropna(axis=0, how='any')
        _X, _y = Xy[X.columns], Xy[y.name]
        # Initialization
        self.index = _X.index
        self.target = _y.name
        self.columns = _X.columns
        self.num_cols = _X.select_dtypes(include='number').columns
        self.cat_cols = _X.select_dtypes(exclude='number').columns
        self.cat_cols = self.columns.drop(self.num_cols)

    def predict(self, x: pd.Series, return_neighbors: bool = False):
        # Compute all pairwise distances
        dists = self.distance(x)
        # Select the k nearest neighbors
        idx = np.argpartition(dists, self.k)[:self.k]
        idx_neighbors = dists.iloc[idx].index
        features_k = self.features.loc[idx_neighbors]
        labels_k = self.labels.loc[idx_neighbors]
        # Majority vote
        label_pred = Counter(labels_k).most_common(1)[0][0]
        # Return class label and/or neighbors
        if return_neighbors:
            neighbors = pd.concat([features_k, labels_k], axis=1)
            return label_pred, neighbors
        else:
            return label_pred

    def impute(self, X: pd.DataFrame) -> pd.DataFrame:
        # Sanity check
        assert all(X.columns == self.columns), "Entries mismatch"
        # Combine X and self.features into the entire dataset
        E = pd.concat([self.features, X])
        # Impute each row of X
        X_imputed = []
        for index, x in X.iterrows():
            # Find k nearest neighbors
            _, neighbors = self.predict(x, return_neighbors=True)
            neighbors.drop(columns=self.target)
            neighbors_num = neighbors[self.num_cols]
            neighbors_cat = neighbors[self.cat_cols]
            # Impute values
            impute_num = neighbors_num.mean()
            impute_cat = neighbors_cat.mode()
            # Breaking ties for categorical values
            if len(impute_cat) > 1:  # at least one entry includes ties
                ties_idx = impute_cat.columns[impute_cat.count() > 1]
                ties = impute_cat[ties_idx]
                # Break ties by comparing occurrences in the entire dataset
                wins = {}
                for tie in ties.iteritems():
                    feature, cat = tie
                    # Filter occurrences of interest
                    cat_counts = E[feature].value_counts()[cat.dropna()]
                    # Select the category with the highest frequency
                    cat_win = cat_counts.sort_values(ascending=False).index[0]
                    # Update impute_cat
                    wins[feature] = cat_win
                # Update and clean up impute_cat
                for feature, cat_win in wins.items():
                    impute_cat.loc[0, feature] = cat_win
            # Combine impute values
            impute_cat = impute_cat.loc[0]  # squeeze impute_cat into pd.Series
            impute_val = pd.concat([impute_num, impute_cat])
            # Fill missing values
            _nan_cols = self.columns[x.isna()]
            x_imputed = x.copy()
            x_imputed[_nan_cols] = impute_val[_nan_cols]
            X_imputed.append(x_imputed)
        # Clean up X_imputed
        X_imputed = pd.DataFrame(X_imputed, index=X.index)
        return X_imputed

    def distance(self, x: pd.Series) -> pd.Series:
        # Sanity check
        assert all(x.index == self.columns), "Entries mismatch"
        # Drop columns with missing values
        _nan_cols = self.columns[x.isna()]
        _num_cols = self.num_cols.drop(_nan_cols, errors='ignore')
        _cat_cols = self.cat_cols.drop(_nan_cols, errors='ignore')
        # Split numerical (continuous) and categorical parts
        x_num = x[_num_cols].to_numpy().reshape(1, -1)
        features_num = self.features[_num_cols].to_numpy()
        x_cat = x[_cat_cols]
        features_cat = self.features[_cat_cols]
        # Compute the distance
        dist_num = cdist(x_num, features_num).squeeze(0)
        dist_cat = np.sum(10 * (x_cat == features_cat), axis=1)
        dist = pd.Series(dist_num + dist_cat, index=self.index)
        return dist
