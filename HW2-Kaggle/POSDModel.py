import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from catboost import CatBoostClassifier
from scipy.spatial.distance import cosine
from tqdm import tqdm  # For progress bar
import random
from sklearn.cluster import KMeans
from typing import List, Tuple
from sklearn.base import ClassifierMixin
# Scikit-learn estimators
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

def compute_coverage(
    model_indices: List[int],
    models: List[ClassifierMixin],
    X_calib: pd.DataFrame,
    y_calib: np.ndarray,
    features_used: List[List[str]]
) -> float:
    """
    Compute fraction of samples correctly predicted (coverage)
    by unweighted ensemble of the given subset of models.

    Args:
        model_indices (List[int]): Indices of models in the subset.
        models (List[ClassifierMixin]): All candidate models.
        X_calib (pd.DataFrame): Calibration features.
        y_calib (np.ndarray): Calibration labels.
        features_used (List[List[str]]): For each model, which features it uses.

    Returns:
        float: Coverage = fraction of y_calib correctly predicted by the subset ensemble.
    """
    if not model_indices:
        return 0.0  # No models => coverage is 0

    # Sum predicted probabilities across the subset
    ensemble_probs = np.zeros(len(X_calib))
    for idx in model_indices:
        feats = features_used[idx]
        ensemble_probs += models[idx].predict_proba(X_calib[feats])[:, 1]

    # Average them (unweighted)
    ensemble_probs /= len(model_indices)

    return roc_auc_score(y_calib, ensemble_probs)



class POSDModel:
    def __init__(self, k=5, t=3, row_subsample=0.8, col_subsample=0.8, iterations=1000, learning_rate=0.05):
        """
        Initialize the POSDModel parameters.
        
        Args:
        k (int): Number of stratified splits for training models.
        t (int): Number of models to select for the final ensemble.
        row_subsample (float): Fraction of rows to subsample for each model.
        col_subsample (float): Fraction of columns to subsample for each model.
        iterations (int): Number of iterations for each CatBoost model.
        learning_rate (float): Learning rate for each CatBoost model.
        """
        self.k = k
        self.t = t
        self.row_subsample = row_subsample
        self.col_subsample = col_subsample
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.models = []
        self.model_weights = []

    def _subsample(self, X, y):
        """Subsample rows and columns."""
        rows_ratio  = random.uniform(self.row_subsample - 0.1, self.row_subsample + 0.1)
        col_ratio = random.uniform(self.col_subsample - 0.1, self.col_subsample + 0.1)
        
        row_indices = np.random.choice(X.index, size=int(len(X) * rows_ratio), replace=False)
        col_indices = np.random.choice(X.columns, size=int(X.shape[1] * col_ratio), replace=False)
        # Create freq based sample weights
        X_sub = X.loc[row_indices, col_indices]
        y_sub = y.loc[row_indices]
        sample_weights_map = y_sub.value_counts(normalize=True)
        sample_weights = y_sub.map(sample_weights_map)
       
        return X_sub, y_sub,sample_weights

    def fit(self, X_train, y_train, calibration_data):
        """
        Train the POSDModel using K stratified splits.
        
        Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        calibration_data (tuple): Data for calibration (X_calib, y_calib).
        """
        X_calib, y_calib = calibration_data
        skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=42)

        model_scores = []
        for train_index, _ in tqdm(skf.split(X_train, y_train), desc="Training models", total=self.k):
            # Stratified split and subsample
            X_split, y_split = X_train.iloc[train_index], y_train.iloc[train_index]
            X_sub, y_sub,sw = self._subsample(X_split, y_split)

            # Choose model type at random for diversity
            model_type = random.choice(["catboost", "rf", "logistic", "knn", "naivebayes","dt"])

            if model_type == "dt":
                model = DecisionTreeClassifier(
                    max_depth=random.randint(3, 10),
                    min_samples_split=random.randint(2, 10),
                    min_samples_leaf=random.randint(1, 5),
                    criterion=random.choice(["gini", "entropy"])
                )
                # Randomly use sample weights
                if random.random() < 0.5:
                    model.fit(X_sub, y_sub)
                else:
                    model.fit(X_sub, y_sub,sample_weight=sw)
                features_used = X_sub.columns.tolist()

            elif model_type == "catboost":
                # CatBoost with random hyperparams
                model = CatBoostClassifier(
                    iterations=self.iterations + random.randint(-self.iterations // 2, self.iterations // 2),
                    learning_rate=self.learning_rate + random.uniform(-0.05, 0.05),
                    depth=random.randint(3, 10),
                    l2_leaf_reg=random.randint(1, 10),
                    loss_function=random.choice(["Logloss", "CrossEntropy"]),
                    verbose=0
                )
                model.fit(X_sub, y_sub)
                # CatBoost can track its own features
                features_used = model.feature_names_

            elif model_type == "rf":
                # RandomForest with random hyperparams
                n_estimators = random.randint(50, 200)
                max_depth = random.randint(3, 10)
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
                model.fit(X_sub, y_sub)
                features_used = X_sub.columns.tolist()

            elif model_type == "logistic":
                # LogisticRegression with random hyperparams
                c_val = 10 ** random.uniform(-2, 2)  # Random log-scale range
                solver_choice = random.choice(["liblinear", "lbfgs"])
                model = LogisticRegression(C=c_val, solver=solver_choice, max_iter=1000)
                model.fit(X_sub, y_sub)
                features_used = X_sub.columns.tolist()

            elif model_type == "knn":
                # KNN with random hyperparams
                n_neighbors = random.randint(3, 15)
                model = KNeighborsClassifier(n_neighbors=n_neighbors)
                model.fit(X_sub, y_sub)
                features_used = X_sub.columns.tolist()

            else:  # "naivebayes"
                # Naive Bayes (Gaussian)
                model = GaussianNB()
                model.fit(X_sub, y_sub)
                features_used = X_sub.columns.tolist()
            
            # Evaluate on calibration set
            y_calib_pred_proba = model.predict_proba(X_calib[features_used])[:, 1]
            auc = roc_auc_score(y_calib, y_calib_pred_proba)
            model_scores.append((model, auc, features_used))
        
        # Select the top T models based on diversity and AUC
        self.models = self._select_top_models_forward_auc(model_scores, X_calib, y_calib)
        self.model_weights = [auc for model, auc ,_ in model_scores if model in self.models]
        self.model_weights = [w / sum(self.model_weights) for w in self.model_weights]
        # plot tiny distribution of models aucs
        plt.hist([auc for model, auc,_ in model_scores], bins=20)
        plt.show()

    def _select_top_models_scores(self, model_scores, X_calib, y_calib):
        """Select the top T models based on a weighted combination of diversity and AUC."""
        models, aucs = zip(*model_scores)
        similarities = []

        # Calculate cosine similarity between model predictions
        for i, model_i in enumerate(models):
            preds_i = model_i.predict_proba(X_calib)[:, 1]
            for j, model_j in enumerate(models):
                if i < j:
                    preds_j = model_j.predict_proba(X_calib)[:, 1]
                    similarity = 1 - cosine(preds_i, preds_j)  # Diversity measure
                    similarities.append((i, j, similarity))

        # Weighted scoring of models
        diversity_weight = 0.5  # Adjust as needed
        auc_weight = 0.5
        scores = []

        for i in range(len(models)):
            diversity_score = np.mean([sim[2] for sim in similarities if sim[0] == i or sim[1] == i])
            combined_score = auc_weight * aucs[i] - diversity_weight * diversity_score
            scores.append((i, combined_score))

        # Select top T models based on combined score
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        selected_indices = [score[0] for score in scores[:self.t]]

        return [models[i] for i in selected_indices]

    def _select_top_models_bandit(
        self,
        model_scores: List[Tuple[ClassifierMixin, float, List[str]]],
        X_calib: pd.DataFrame,
        y_calib: np.ndarray
    ) -> List[ClassifierMixin]:
        """
        Use a UCB bandit approach to iteratively pick models that yield 
        the largest coverage improvement, balancing exploration & exploitation.

        Args:
            model_scores (List[Tuple[ClassifierMixin, float, List[str]]]): 
                Each tuple is (model, model_auc, features_used).
            X_calib (pd.DataFrame):
                Calibration features.
            y_calib (np.ndarray):
                Calibration target values.

        Returns:
            List[ClassifierMixin]:
                A list of the selected top models (size up to self.t).
        """
        # Unpack the model info
        models, aucs, features_used = zip(*model_scores)
        
        # 1) Filter by an AUC threshold to get candidate "arms"
        auc_threshold = 0.75
        eligible_indices = [i for i in range(len(models)) if aucs[i] >= auc_threshold]
        print(f"Number of eligible models: {len(eligible_indices)}")
        if len(eligible_indices) == 0:
            raise ValueError("No models meet the AUC threshold.")

        # 2) Initialize empty subset, bandit stats
        selected_indices: List[int] = []
        coverage_current = 0.0
        counts = np.zeros(len(eligible_indices), dtype=int)    # times each arm was pulled
        values = np.zeros(len(eligible_indices), dtype=float)  # average reward (incremental coverage)
        alpha = 1.5  # UCB exploration constant

        # Helper function to compute coverage for a subset of model indices
        def coverage_of_subset(subset: List[int]) -> float:
            return compute_coverage(subset, models, X_calib, y_calib, features_used)

        # 3) Optionally pick the single best model by AUC as a starter
        best_single_idx = max(eligible_indices, key=lambda idx: aucs[idx])
        selected_indices.append(best_single_idx)
        coverage_current = coverage_of_subset([best_single_idx])
        print(f"Selected first model: {best_single_idx} "
            f"with AUC: {aucs[best_single_idx]:.3f} - coverage {coverage_current:.3f}")

        # 4) Bandit loop: pick up to self.t - 1 additional models
        while len(selected_indices) < self.t:
            print(f"Current subset: {selected_indices}")
            unselected_arms = [arm for arm in eligible_indices if arm not in selected_indices]

            # If no unselected arms remain, break
            if not unselected_arms:
                print("No more arms left to try.")
                break

            # We'll attempt one arm each iteration (the best by UCB) and see if coverage improves
            improvement_found = False

            # Calculate UCB for each candidate arm
            total_pulls = counts.sum() + 1e-9  # avoid div by zero
            ucb_values = []

            for arm_i, arm in enumerate(eligible_indices):
                if arm in selected_indices:
                    ucb_values.append(-np.inf)  # already in the subset, skip
                    continue

                if counts[arm_i] == 0:
                    # Force exploration of never-pulled arms
                    ucb_values.append(float('inf'))
                else:
                    # UCB formula
                    exploit = values[arm_i]
                    explore = alpha * np.sqrt(np.log(total_pulls) / counts[arm_i])
                    ucb_values.append(exploit + explore)

            # Pick the best arm
            best_arm_index = int(np.argmax(ucb_values))
            best_arm = eligible_indices[best_arm_index]

            # 5) Evaluate coverage improvement (reward)
            new_subset = selected_indices + [best_arm]
            new_coverage = coverage_of_subset(new_subset)
            reward = new_coverage - coverage_current

            # Update bandit counts & average value
            counts[best_arm_index] += 1
            old_val = values[best_arm_index]
            n = counts[best_arm_index]
            values[best_arm_index] = old_val + (reward - old_val) / n

            # 6) If improvement is positive, accept that arm
            if reward > 0:
                selected_indices.append(best_arm)
                coverage_current = new_coverage
                improvement_found = True
                print(f"Added model {best_arm} => coverage improved by {reward:.4f} "
                    f"to {coverage_current:.4f}")
            else:
                # If reward <= 0, optionally penalize it so we don't pick it repeatedly
                # For instance, subtract 0.5 or 1.0 from its value
                values[best_arm_index] -= 1.0
                print(f"Arm {best_arm} gave no improvement. Penalizing its UCB for next iteration.")

            # Stop if we've used up all arms or found T models
            if len(selected_indices) == len(eligible_indices):
                break

            # If you want to stop as soon as there's no improvement from any unselected arms,
            # you'll need a bigger loop over *all* unselected arms in a single iteration,
            # or track if we tried them all. For now, we break if no improvement from the best arm:
            if not improvement_found:
                print("No improvement from best arm => stopping bandit selection.")
                break

        # Step 7: Print final coverage
        print(f"Final subset coverage: {coverage_current*100:.2f}%")

        # Store selected model features for subsequent usage
        self.selected_model_features = [features_used[i] for i in selected_indices]

        # Return the chosen models
        return [models[i] for i in selected_indices]

    
    def _select_top_models_clusters(self, model_scores, X_calib, y_calib):
        """Cluster models based on predictions and select one from each cluster."""
        models, aucs = zip(*model_scores)

        # Create a matrix of prediction probabilities
        prediction_matrix = np.array([model.predict_proba(X_calib)[:, 1] for model in models]).T

        # Cluster models using K-Means
        n_clusters = self.t
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(prediction_matrix.T)

        # Select the model with the highest AUC from each cluster
        selected_models = []
        for cluster in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster)[0]
            best_model_index = max(cluster_indices, key=lambda idx: aucs[idx])
            selected_models.append(models[best_model_index])

        return selected_models
    def _select_top_models_regular(self, model_scores, X_calib, y_calib):
        """Prioritize models with high AUC and regularize diversity."""
        models, aucs = zip(*model_scores)
        similarities = []

        # Calculate cosine similarity between model predictions
        for i, model_i in enumerate(models):
            preds_i = model_i.predict_proba(X_calib)[:, 1]
            for j, model_j in enumerate(models):
                if i < j:
                    preds_j = model_j.predict_proba(X_calib)[:, 1]
                    similarity = 1 - cosine(preds_i, preds_j)  # Diversity measure
                    similarities.append((i, j, similarity))

        # Filter models with AUC above a threshold
        auc_threshold = 0.83  # Set based on dataset
        eligible_models = [(i, aucs[i]) for i in range(len(models)) if aucs[i] >= auc_threshold]
        print(f'Number of eligible models: {len(eligible_models)}')
        # Sort by diversity among eligible models
        selected_models = set()
        for i, j, sim in sorted(similarities, key=lambda x: x[2]):  # Low similarity first
            if i in [m[0] for m in eligible_models] and len(selected_models) < self.t:
                selected_models.add(i)
            if j in [m[0] for m in eligible_models] and len(selected_models) < self.t:
                selected_models.add(j)
            if len(selected_models) >= self.t:
                break

        return [models[i] for i in selected_models]

    


    def _select_top_models_forward_auc(
        self,
        model_scores: List[Tuple[ClassifierMixin, float, List[str]]],
        X_calib: pd.DataFrame,
        y_calib: np.ndarray
    ) -> List[ClassifierMixin]:
        """
        Forward selection of models to maximize ensemble AUC. At each step,
        add the model that yields the greatest AUC improvement, until no
        improvement or we reach self.t models.

        Args:
            model_scores: List of (model, model_auc, features_used).
            X_calib, y_calib: Calibration data + labels.

        Returns:
            A list of the selected top models (size up to self.t).
        """
        models, aucs, features_used = zip(*model_scores)

        # Convert to a simple index list for convenience
        candidate_indices = list(range(len(models)))

        # Optional: You can filter by an AUC threshold if desired:
        # e.g. candidate_indices = [i for i in candidate_indices if aucs[i] >= 0.75]

        # We'll store which models we've chosen
        selected_indices: List[int] = []

        # Optionally, pick the best single model by AUC to start
        best_single_idx = max(candidate_indices, key=lambda i: aucs[i])
        selected_indices.append(best_single_idx)

        # Compute initial ensemble AUC
        ensemble_probs = compute_ensemble_probs(selected_indices, models, X_calib, features_used)
        current_auc = roc_auc_score(y_calib, ensemble_probs)
        print(f"Starting with model {best_single_idx}, individual AUC={aucs[best_single_idx]:.3f}, "
            f"ensemble AUC={current_auc:.3f}")

        # While we have not reached t models, try to add the next best improvement
        while len(selected_indices) < self.t:
            best_gain = 0.0
            best_model_to_add = None

            # Check each candidate not in selected_indices
            for i in candidate_indices:
                if i in selected_indices:
                    continue
                # Compute the new ensemble's AUC if we add model i
                trial_subset = selected_indices + [i]
                trial_probs = compute_ensemble_probs(trial_subset, models, X_calib, features_used)
                trial_auc = roc_auc_score(y_calib, trial_probs)

                gain = trial_auc - current_auc
                if gain > best_gain:
                    best_gain = gain
                    best_model_to_add = i

            # If no model yields a positive gain, stop
            if best_gain <= 0:
                print("No further improvement in AUC => stopping.")
                break

            # Otherwise, add the best model to the subset
            selected_indices.append(best_model_to_add)
            current_auc += best_gain  # or just reassign current_auc = trial_auc
            print(f"Added model {best_model_to_add} => AUC improved by {best_gain:.4f}, "
                f"ensemble AUC now {current_auc:.4f}")

            # If we've used up all candidates, break
            if len(selected_indices) == len(candidate_indices):
                break

        print(f"Final ensemble of {len(selected_indices)} models with AUC={current_auc:.3f}")

        # Track selected model features for subsequent usage
        self.selected_model_features = [features_used[i] for i in selected_indices]

        # Return the chosen models
        return [models[i] for i in selected_indices]

    def _select_top_models(
        self,
        model_scores: List[Tuple[ClassifierMixin, float, List[str]]],
        X_calib: pd.DataFrame,
        y_calib: np.ndarray
    ) -> List[ClassifierMixin]:
        """
        Prioritize models with high AUC and complementarity based on error correction.
        Only the features actually used by each model are passed in for calibration predictions.

        Args:
            model_scores (List[Tuple[ClassifierMixin, float, List[str]]]): 
                A list where each tuple is (model, model_auc, model_features_used).
            X_calib (pd.DataFrame):
                Calibration features.
            y_calib (np.ndarray):
                Calibration target values.

        Returns:
            List[ClassifierMixin]:
                A list of the selected top models.
        """
        models, aucs, features_used = zip(*model_scores)

        # Step 1: Filter models based on AUC threshold
        auc_threshold = 0.84  # Set based on dataset
        eligible_models = [(i, aucs[i]) for i in range(len(models)) if aucs[i] >= auc_threshold]
        print(f"Number of eligible models: {len(eligible_models)}")

        if len(eligible_models) == 0:
            raise ValueError("No models meet the AUC threshold.")

        # Step 2: Select the best model by AUC as the starting point
        selected_indices = []
        first_model_index = max(eligible_models, key=lambda x: x[1])[0]
        selected_indices.append(first_model_index)
        print(f"Selected first model: {first_model_index} with AUC: {aucs[first_model_index]}")

        # Step 3: Iteratively select the most complementary model
        covered_samples = set()
        for model_idx in selected_indices:
            current_feats = features_used[model_idx]
            preds = (models[model_idx].predict_proba(X_calib[current_feats])[:, 1] > 0.5).astype(int)
            covered_samples.update(np.where(preds == y_calib)[0])

        while len(selected_indices) < self.t and len(covered_samples) < len(y_calib):
            best_additional_model = None
            max_new_coverage = 0

            for i, auc in eligible_models:
                if i in selected_indices:
                    continue
                current_feats = features_used[i]
                preds = (models[i].predict_proba(X_calib[current_feats])[:, 1] > 0.5).astype(int)
                correct_indices = set(np.where(preds == y_calib)[0])
                new_coverage = len(correct_indices - covered_samples)

                if new_coverage > max_new_coverage:
                    max_new_coverage = new_coverage
                    best_additional_model = i

            if best_additional_model is not None:
                selected_indices.append(best_additional_model)
                current_feats = features_used[best_additional_model]
                preds = (models[best_additional_model].predict_proba(X_calib[current_feats])[:, 1] > 0.5).astype(int)
                covered_samples.update(np.where(preds == y_calib)[0])
                print(
                    f"Selected model: {best_additional_model} with AUC: {aucs[best_additional_model]} "
                    f"and new coverage: {max_new_coverage}"
                )
            else:
                break

        # Step 4: Compute and print total coverage
        total_coverage = len(covered_samples) / len(y_calib) * 100
        print(f"Total coverage achieved: {total_coverage:.2f}%")

        # Track selected model features for subsequent usage
        self.selected_model_features = [features_used[i] for i in selected_indices]

        # Return selected models
        return [models[i] for i in selected_indices]


    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Predict using the ensemble of top T models.
        Only the features actually used by each model are passed in for predictions.

        Args:
            X_test (pd.DataFrame):
                Test features.

        Returns:
            np.ndarray:
                Final binary predictions.
        """
        ensemble_preds = np.zeros(len(X_test))

        # Ensure self.selected_model_features is aligned with self.models
        for model, weight, feats in zip(self.models, self.model_weights, self.selected_model_features):
            ensemble_preds += weight * model.predict_proba(X_test[feats])[:, 1]

        return (ensemble_preds > 0.5).astype(int)


    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities using the ensemble of top T models.
        Only the features actually used by each model are passed in for predictions.

        Args:
            X_test (pd.DataFrame):
                Test features.

        Returns:
            np.ndarray:
                Final probability predictions as a 1D array.
        """
        ensemble_preds = np.zeros(len(X_test))

        # Ensure self.selected_model_features is aligned with self.models
        for model, weight, feats in zip(self.models, self.model_weights, self.selected_model_features):
            ensemble_preds += weight * model.predict_proba(X_test[feats])[:, 1]

        return ensemble_preds
    
def compute_ensemble_probs(
        subset_indices: List[int],
        models: List[ClassifierMixin],
        X_calib: pd.DataFrame,
        features_used: List[List[str]]
    ) -> np.ndarray:
        """
        Compute the unweighted average probability of class=1 for 
        the subset of models in 'subset_indices'.
        """
        if not subset_indices:
            # No models => default to probability=0 for all
            return np.zeros(len(X_calib))

        # Sum up predicted probabilities from each model in the subset
        ensemble_probs = np.zeros(len(X_calib))
        for idx in subset_indices:
            feats = features_used[idx]
            ensemble_probs += models[idx].predict_proba(X_calib[feats])[:, 1]

        # Take the average
        ensemble_probs /= len(subset_indices)
        return ensemble_probs

# Example usage
# X_train,y_train = balance_classes(X_train_no_outliers, y_train_no_outliers, method='over')
X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(X_train, y_train, test_size=0.01)
posd_model = POSDModel(k=1000, t=50, row_subsample=0.1, col_subsample=0.2, iterations=120, learning_rate=0.1)
posd_model.fit(X_train_res, y_train_res, calibration_data=(X_test_res, y_test_res))

y_pred = posd_model.predict(X_test)
print(f"Final Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Final AUC: {roc_auc_score(y_test, posd_model.predict_proba(X_test))}")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

def identify_bad_cases(
    posd_model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_subsamples: int = 200,
    subsample_size: int = 200,
    auc_threshold: float = 0.80
) -> List[int]:
    """
    Identify indices of 'bad cases' from repeated subsampling, i.e.,
    samples that appear in runs where the AUC < auc_threshold.

    Args:
        posd_model: A trained ensemble model with predict_proba().
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True labels for X_test.
        n_subsamples (int): Number of subsampling iterations.
        subsample_size (int): Size of each subsample drawn from X_test.
        auc_threshold (float): Subsamples with AUC below this value are considered "bad runs."

    Returns:
        List[int]: A list of sample indices that were part of "bad runs."
    """
    bad_indices = []

    for i in range(n_subsamples):
        sub_test = X_test.sample(subsample_size, replace=False)
        sub_y = y_test.loc[sub_test.index]
        
        # Predict probabilities on this subsample
        sub_proba = posd_model.predict_proba(sub_test)  # shape (N,), or (N, ) for binary
        cur_auc = roc_auc_score(sub_y, sub_proba)
        
        if cur_auc < auc_threshold:
            # If this run is "bad," record all indices from this subsample
            bad_indices.extend(list(sub_test.index))
    
    return bad_indices


def cluster_bad_cases(
    X_test: pd.DataFrame,
    bad_indices: List[int],
    n_clusters: int = 3
) -> Tuple[KMeans, pd.DataFrame]:
    """
    Cluster the 'bad' samples using K-Means, returning the unscaled DataFrame
    with a new 'cluster_id' column.

    Args:
        X_test (pd.DataFrame): Full test features.
        bad_indices (List[int]): Indices of 'bad' samples.
        n_clusters (int): Number of clusters to use for K-Means.

    Returns:
        Tuple[KMeans, pd.DataFrame]: (trained KMeans model, DataFrame of bad samples + cluster_id).
    """
    # 1) Subset the DataFrame to just the 'bad' samples (unscaled copy)
    bad_samples_df = X_test.loc[bad_indices].copy()

    # 2) Select only numeric columns for clustering
    numeric_cols = bad_samples_df.select_dtypes(include=[np.number]).columns.tolist()
    # If you want to exclude certain columns (like 'id'), remove them here
    # numeric_cols = [col for col in numeric_cols if col != 'id']

    # 3) Create a copy that we will scale
    X_bad_numeric = bad_samples_df[numeric_cols].copy()

    # 4) Handle missing values if needed (fill or drop)
    X_bad_numeric = X_bad_numeric.fillna(0)

    # 5) Scale the numeric data
    scaler = StandardScaler()
    X_bad_scaled = scaler.fit_transform(X_bad_numeric)

    # 6) Fit and predict clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_ids = kmeans.fit_predict(X_bad_scaled)

    # 7) Attach the cluster labels back to the unscaled DataFrame
    bad_samples_df['cluster_id'] = cluster_ids

    return kmeans, bad_samples_df


def build_correction_model(
    posd_model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    bad_samples_df: pd.DataFrame
) -> LogisticRegression:
    """
    Train a small 'correction' model that uses the ensemble's predicted probability
    and original features to predict the true label, focusing on the bad samples.

    Args:
        posd_model: The main ensemble model (should have predict_proba()).
        X_test (pd.DataFrame): Full test features.
        y_test (pd.Series): True labels for X_test.
        bad_samples_df (pd.DataFrame): The subset of samples identified as "bad cases,"
            with a 'cluster_id' column from the clustering step.

    Returns:
        LogisticRegression: A trained logistic regression correction model.
    """
    # Collect features + ensemble predictions for bad cases
    bad_indices = bad_samples_df.index
    X_bad = X_test.loc[bad_indices].copy()
    y_bad = y_test.loc[bad_indices]
    
    # The ensemble's predicted probability
    ensemble_probs = posd_model.predict_proba(X_bad)
    X_bad['ensemble_proba'] = ensemble_probs

    # Optionally, add the cluster ID as a feature
    X_bad['cluster_id'] = bad_samples_df.loc[bad_indices, 'cluster_id'].values

    # Train a small logistic regression to correct the final prediction
    correction_model = LogisticRegression(max_iter=1000)
    correction_model.fit(X_bad, y_bad)

    return correction_model


def corrected_predict_proba(
    posd_model,
    correction_model: LogisticRegression,
    kmeans: KMeans,
    X_new: pd.DataFrame
) -> np.ndarray:
    """
    Predict probabilities using the ensemble's predictions, and apply a correction
    based on cluster membership and a logistic regression correction model.

    Args:
        posd_model: The main ensemble model (predict_proba()).
        correction_model (LogisticRegression): The trained correction model.
        kmeans (KMeans): Trained K-Means model for identifying clusters.
        X_new (pd.DataFrame): New data samples for prediction.

    Returns:
        np.ndarray: Corrected probability predictions as a 1D array.
    """
    # Predict raw ensemble probabilities
    ensemble_probs = posd_model.predict_proba(X_new)
    
    # Identify cluster for each new sample
    cluster_ids = kmeans.predict(X_new)
    
    # Build features for correction model
    X_correct = X_new.copy()
    X_correct['ensemble_proba'] = ensemble_probs
    X_correct['cluster_id'] = cluster_ids
    
    # Correction model output is probability that the label = 1
    corrected_probs = correction_model.predict_proba(X_correct)[:, 1]
    return corrected_probs


import numpy as np
import random
from typing import List, Callable

def coverage_metric(y_true: np.ndarray, preds: np.ndarray) -> float:
    """
    Example coverage metric: fraction of samples for which preds == y_true.
    y_true, preds are both 0/1 arrays of the same shape.
    """
    # Should be auc
    return roc_auc_score(y_true, preds)

def ensemble_predict(
    models: List,  # subset of selected models
    X_calib, 
    weights: List[float] = None
) -> np.ndarray:
    """
    Simple ensemble prediction: average predicted probability from all models, threshold at 0.5.
    If weights is None, all models have equal weight.
    """
    if not models:
        return np.zeros(len(X_calib), dtype=int)  # no models => all zeros

    if weights is None:
        weights = [1.0 / len(models)] * len(models)

    # Weighted sum of predicted probabilities
    ensemble_probs = np.zeros(len(X_calib))
    for m, w in zip(models, weights):
        ensemble_probs += w * m.predict_proba(X_calib)[:, 1]
    final_preds = (ensemble_probs >= 0.5).astype(int)
    return final_preds

class BanditModelSelector:
    def __init__(
        self,
        all_models: List,            # Large pool of candidate models
        X_calib, y_calib,           # Calibration data
        metric_func: Callable,      # e.g. coverage_metric or AUC-based
        max_subset_size: int = 5,
        ucb_alpha: float = 2.0
    ):
        self.all_models = all_models
        self.X_calib = X_calib
        self.y_calib = y_calib
        self.metric_func = metric_func
        self.max_subset_size = max_subset_size
        self.ucb_alpha = ucb_alpha

        # Bandit stats
        self.counts = np.zeros(len(all_models), dtype=int)   # how many times each arm was pulled
        self.values = np.zeros(len(all_models), dtype=float) # average reward for each arm

        # Subset of chosen models
        self.selected_indices = []

        # Current performance of the ensemble
        self.current_score = 0.0

    def _calculate_reward(self, arm_index: int) -> float:
        """
        Compute incremental improvement by adding 'arm_index' model to the current ensemble subset.
        """
        new_models = [self.all_models[i] for i in self.selected_indices + [arm_index]]
        preds = ensemble_predict(new_models, self.X_calib)
        new_score = self.metric_func(self.y_calib, preds)
        reward = new_score - self.current_score
        return reward

    def select_models(self):
        """
        Use a UCB-based approach to pick up to 'max_subset_size' models.
        """
        while len(self.selected_indices) < self.max_subset_size:
            # If we haven't tried all arms at least once, pick those first
            untried_arms = [i for i in range(len(self.all_models)) 
                            if i not in self.selected_indices and self.counts[i] == 0]
            if untried_arms:
                arm = random.choice(untried_arms)
            else:
                # Compute UCB for each candidate arm
                # Only consider arms not yet in selected_indices
                ucb_values = []
                for i in range(len(self.all_models)):
                    if i in self.selected_indices:
                        ucb_values.append(-np.inf)  # exclude
                        continue
                    exploration = self.ucb_alpha * np.sqrt(np.log(sum(self.counts)) / self.counts[i])
                    ucb_values.append(self.values[i] + exploration)
                arm = int(np.argmax(ucb_values))

            # Evaluate reward
            reward = self._calculate_reward(arm)
            self.counts[arm] += 1
            # Update average reward for that arm
            n = self.counts[arm]
            old_val = self.values[arm]
            self.values[arm] = old_val + (reward - old_val)/n

            # If reward > 0 => it improves the ensemble, add it
            if reward > 0:
                self.selected_indices.append(arm)
                self.current_score += reward
            else:
                # If no improvement, we skip adding that model
                # or possibly break if we want to stop early
                pass
        
        return self.selected_indices

# Usage Example:
# Suppose we have "candidate_models = [model1, model2, ...]" and "X_calib, y_calib".
# We'll do coverage-based selection with a maximum subset size of 5.

# bandit_selector = BanditModelSelector(
#     all_models=candidate_models,
#     X_calib=X_calib,
#     y_calib=y_calib,
#     metric_func=coverage_metric,  # or AUC-based
#     max_subset_size=5,
#     ucb_alpha=2.0
# )
# chosen_indices = bandit_selector.select_models()
# final_subset = [candidate_models[i] for i in chosen_indices]
