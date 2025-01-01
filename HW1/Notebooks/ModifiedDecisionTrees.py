import numpy as np
from sklearn.tree import DecisionTreeClassifier

class SoftSplitDecisionTreeClassifierImproved(DecisionTreeClassifier):
    def __init__(self, alpha: float = 0.1, n_simulations: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.n_simulations = n_simulations

    def _distance_from_uniform(self, class_counts: np.ndarray) -> float:
        """
        Calculate the distance of the class distribution from a uniform distribution.
        """
        n_classes = len(class_counts)
        uniform_dist = np.full(n_classes, 1 / n_classes)  # Uniform distribution
        class_proba = class_counts / class_counts.sum()  # Normalize to probabilities

        # Compute distance (e.g., KL Divergence)
        kl_divergence = np.sum(class_proba * np.log(class_proba / uniform_dist + 1e-8))
        return kl_divergence

    def _soft_split(self, feature_index: int, threshold: float, sample: np.ndarray, tree, node) -> int:
        """
        Perform a soft split with weighting based on distance from uniformity.
        """
        # Calculate distance from uniform for this node
        class_counts = tree.value[node][0]
        distance = self._distance_from_uniform(class_counts)

        # Adjust alpha based on distance
        adjusted_alpha = self.alpha / (1 + distance)

        # Probability to go right
        prob_go_right = (
            1 - adjusted_alpha if sample[feature_index] > threshold else adjusted_alpha
        )
        return np.random.choice([1, 0], p=[prob_go_right, 1 - prob_go_right])

    def _predict_sample_proba(self, tree, sample: np.ndarray) -> np.ndarray:
        node = 0  # Start at the root node
        while tree.children_left[node] != -1:  # While it's not a leaf node
            feature_index = tree.feature[node]
            threshold = tree.threshold[node]
            child_direction = self._soft_split(feature_index, threshold, sample, tree, node)
            node = (
                tree.children_left[node]
                if child_direction == 0
                else tree.children_right[node]
            )
        # Normalize leaf node's class counts to get probabilities
        return tree.value[node][0] / tree.value[node][0].sum()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)  # Ensure X is a numpy array
        tree = self.tree_
        n_classes = len(self.classes_)
        probabilities = []

        for sample in X:
            sample = np.asarray(sample)  # Ensure sample is a numpy array
            sample_probas = np.zeros(n_classes)
            for _ in range(self.n_simulations):
                sample_probas += self._predict_sample_proba(tree, sample)
            sample_probas /= self.n_simulations
            probabilities.append(sample_probas)

        return np.array(probabilities)

from sklearn.tree import DecisionTreeClassifier
from scipy.special import softmax
import numpy as np

class LegalAttentionDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self, attention_temperature=1.0, **kwargs):
        """
        Args:
            attention_temperature (float): Controls the sharpness of attention weights.
        """
        super().__init__(**kwargs)
        self.attention_temperature = attention_temperature

    def _attention_weights(self, tree, reachable_leaves):
        """
        Compute attention weights for reachable leaves based on leaf size and entropy.
        """
        # Compute leaf sizes and normalize
        leaf_sizes = np.array([tree.n_node_samples[leaf] for leaf in reachable_leaves])
        leaf_probs = np.array(
            [tree.value[leaf][0] / tree.value[leaf][0].sum() for leaf in reachable_leaves]
        )
        entropies = np.array([-(p * np.log(p + 1e-8)).sum() for p in leaf_probs])

        # Compute scores: larger leaves and lower entropy get higher attention
        scores = np.log(leaf_sizes + 1e-8) - entropies / self.attention_temperature

        # Apply softmax to normalize into attention weights
        return softmax(scores)

    def predict_proba(self, X):
        """
        Predict probabilities using attention-weighted aggregation across leaves.
        """
        tree = self.tree_
        probabilities = []

        for sample in X:
            node = 0  # Start at the root node
            while tree.children_left[node] != -1:  # Traverse until reaching a leaf node
                feature_index = tree.feature[node]
                threshold = tree.threshold[node]
                if sample[feature_index] <= threshold:
                    node = tree.children_left[node]
                else:
                    node = tree.children_right[node]

            # Collect reachable leaves
            reachable_leaves = []
            stack = [node]

            while stack:
                current = stack.pop()
                if tree.children_left[current] == -1:  # If leaf node
                    reachable_leaves.append(current)
                else:
                    stack.append(tree.children_left[current])
                    stack.append(tree.children_right[current])

            # Compute attention-weighted probabilities
            weights = self._attention_weights(tree, reachable_leaves)
            leaf_probs = np.array(
                [tree.value[leaf][0] / tree.value[leaf][0].sum() for leaf in reachable_leaves]
            )
            combined_proba = np.sum(weights[:, None] * leaf_probs, axis=0)
            probabilities.append(combined_proba)

        probabilities = np.array(probabilities)
        return probabilities / probabilities.sum(axis=1, keepdims=True)


class EntropyRegularizedDecisionTreeClassifier(DecisionTreeClassifier):
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        tree = self.tree_
        probabilities = []

        for sample in X:
            node = 0  # Start at the root node
            while tree.children_left[node] != -1:  # Traverse until reaching a leaf node
                feature_index = tree.feature[node]
                threshold = tree.threshold[node]
                if sample[feature_index] <= threshold:
                    node = tree.children_left[node]
                else:
                    node = tree.children_right[node]

            # Get leaf probabilities
            leaf_proba = tree.value[node][0] / tree.value[node][0].sum()
            # print(f"Leaf proba: {leaf_proba}")
            # Regularize by entropy
            leaf_entropy = entropy(leaf_proba)
            regularized_proba = leaf_proba * (1 / (1 + leaf_entropy))
            # print(f"Regularized proba: {regularized_proba}")
            probabilities.append(regularized_proba)

        probabilities = np.array(probabilities)
        return probabilities / probabilities.sum(axis=1, keepdims=True)
    
class ClassSpecificWeightedDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self, class_weights: dict = None, **kwargs):
        """
        Args:
            class_weights (dict): A dictionary mapping class labels to their weights.
        """
        super().__init__(**kwargs)
        self.class_weights = class_weights or {}

    def _weighted_leaf_proba(self, tree, node):
        leaf_proba = tree.value[node][0] / tree.value[node][0].sum()
        class_weighted_proba = [
            leaf_proba[i] * self.class_weights.get(i, 1.0) for i in range(len(leaf_proba))
        ]
        return np.array(class_weighted_proba)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        tree = self.tree_
        probabilities = []

        for sample in X:
            node = 0  # Start at the root node
            while tree.children_left[node] != -1:  # While not a leaf node
                feature_index = tree.feature[node]
                threshold = tree.threshold[node]
                if sample[feature_index] <= threshold:
                    node = tree.children_left[node]
                else:
                    node = tree.children_right[node]

            probabilities.append(self._weighted_leaf_proba(tree, node))

        probabilities = np.array(probabilities)
        return probabilities / probabilities.sum(axis=1, keepdims=True)
    
from scipy.stats import entropy

class EntropyWeightedDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _weighted_leaf_proba(self, tree, node):
        leaf_proba = tree.value[node][0] / tree.value[node][0].sum()
        leaf_entropy = entropy(leaf_proba)  # Calculate Shannon entropy
        weight = 1 / (1 + leaf_entropy)  # Inverse weighting by entropy
        return leaf_proba * weight

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        tree = self.tree_
        probabilities = []

        for sample in X:
            node = 0  # Start at the root node
            while tree.children_left[node] != -1:  # Traverse until reaching a leaf node
                feature_index = tree.feature[node]
                threshold = tree.threshold[node]
                if sample[feature_index] <= threshold:
                    node = tree.children_left[node]
                else:
                    node = tree.children_right[node]

            probabilities.append(self._weighted_leaf_proba(tree, node))

        probabilities = np.array(probabilities)
        return probabilities / probabilities.sum(axis=1, keepdims=True)
    
from sklearn.neighbors import KernelDensity

from sklearn.neighbors import KernelDensity

class KDEWeightedDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self, bandwidth=1.0, **kwargs):
        """
        Args:
            bandwidth (float): Bandwidth for the KDE.
        """
        super().__init__(**kwargs)
        self.bandwidth = bandwidth
        self._X_train = None
        self._y_train = None

    def fit(self, X, y, sample_weight=None):
        """
        Override fit method to store training data.
        """
        self._X_train = X
        self._y_train = y
        super().fit(X, y, sample_weight)
        return self

    def _kde_weighted_proba(self, X_leaf, y_leaf):
        """
        Compute KDE-based probabilities for each class in the leaf.
        """
        kde = KernelDensity(bandwidth=self.bandwidth).fit(X_leaf)
        densities = kde.score_samples(X_leaf)
        weights = np.exp(densities)
        class_probs = {cls: 0 for cls in np.unique(self._y_train)}
        
        for cls in np.unique(y_leaf):
            class_mask = y_leaf == cls
            class_probs[cls] = weights[class_mask].sum()

        # Normalize to ensure probabilities sum to 1
        total_weight = sum(class_probs.values())
        
        return np.array([class_probs[cls] / total_weight for cls in sorted(class_probs)])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities using KDE for each test sample.
        """
        tree = self.tree_
        probabilities = []

        for sample in X:
            node = 0  # Start at the root node
            while tree.children_left[node] != -1:  # Traverse until reaching a leaf node
                feature_index = tree.feature[node]
                threshold = tree.threshold[node]
                if sample[feature_index] <= threshold:
                    node = tree.children_left[node]
                else:
                    node = tree.children_right[node]

            # Get samples in the same leaf node
            leaf_samples = self.apply(self._X_train) == node
            X_leaf = self._X_train[leaf_samples]
            y_leaf = self._y_train[leaf_samples]

            probabilities.append(self._kde_weighted_proba(X_leaf, y_leaf))

        probabilities = np.array(probabilities)
        return probabilities / probabilities.sum(axis=1, keepdims=True)


class BayesianWeightedDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self, prior_probs: dict = None, **kwargs):
        """
        Args:
            prior_probs (dict): A dictionary mapping class labels to prior probabilities.
        """
        super().__init__(**kwargs)
        self.prior_probs = prior_probs or {}

    def _posterior_proba(self, tree, node):
        leaf_counts = tree.value[node][0]
        total_counts = leaf_counts.sum()
        prior = np.array([self.prior_probs.get(i, 1.0) for i in range(len(leaf_counts))])
        likelihood = leaf_counts / total_counts
        posterior = likelihood * prior
        return posterior / posterior.sum()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        tree = self.tree_
        probabilities = []

        for sample in X:
            node = 0  # Start at the root node
            while tree.children_left[node] != -1:  # While not a leaf node
                feature_index = tree.feature[node]
                threshold = tree.threshold[node]
                if sample[feature_index] <= threshold:
                    node = tree.children_left[node]
                else:
                    node = tree.children_right[node]

            probabilities.append(self._posterior_proba(tree, node))

        probabilities = np.array(probabilities)
        return probabilities / probabilities.sum(axis=1, keepdims=True)


class AdaptiveBoostingWeightedDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self, boosting_weights: dict = None, **kwargs):
        """
        Args:
            boosting_weights (dict): A dictionary mapping leaf nodes to their boosting weights.
        """
        super().__init__(**kwargs)
        self.boosting_weights = boosting_weights or {}

    def _boosted_leaf_proba(self, tree, node):
        leaf_proba = tree.value[node][0] / tree.value[node][0].sum()
        weight = self.boosting_weights.get(node, 1.0)
        return leaf_proba * weight

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        tree = self.tree_
        probabilities = []

        for sample in X:
            node = 0  # Start at the root node
            while tree.children_left[node] != -1:  # While not a leaf node
                feature_index = tree.feature[node]
                threshold = tree.threshold[node]
                if sample[feature_index] <= threshold:
                    node = tree.children_left[node]
                else:
                    node = tree.children_right[node]

            probabilities.append(self._boosted_leaf_proba(tree, node))

        probabilities = np.array(probabilities)
        return probabilities / probabilities.sum(axis=1, keepdims=True)


class DepthWeightedDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_depth_weights(self, tree):
        depths = {}
        stack = [(0, 0)]  # (node_id, current_depth)

        while stack:
            node, depth = stack.pop()
            depths[node] = depth
            if tree.children_left[node] != -1:  # If not a leaf node
                stack.append((tree.children_left[node], depth + 1))
                stack.append((tree.children_right[node], depth + 1))

        # Assign weights: weight = 1 / (depth + 1)
        weights = {node: 1 / (depth + 1) for node, depth in depths.items()}
        return weights

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        tree = self.tree_
        depth_weights = self._get_depth_weights(tree)  # Compute depth weights
        probabilities = []

        for sample in X:
            node = 0  # Start at the root node
            while tree.children_left[node] != -1:  # Traverse until reaching a leaf node
                feature_index = tree.feature[node]
                threshold = tree.threshold[node]
                if sample[feature_index] <= threshold:
                    node = tree.children_left[node]
                else:
                    node = tree.children_right[node]

            # Get probabilities from the leaf node and weight by depth
            leaf_proba = tree.value[node][0] / tree.value[node][0].sum()
            weight = depth_weights[node]
            probabilities.append(leaf_proba * weight)

        # Normalize probabilities across all leaves for each sample
        probabilities = np.array(probabilities)
        return probabilities / probabilities.sum(axis=1, keepdims=True)


class WeightedLeafDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _weighted_leaf_proba(self, tree, node):
        # Probability distribution of the leaf
        leaf_proba = tree.value[node][0] / tree.value[node][0].sum()
        # Weight based on the sample density
        weight = tree.value[node][0].sum()
        return leaf_proba * weight

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        tree = self.tree_
        probabilities = []

        for sample in X:
            node = 0  # Start at the root node
            while tree.children_left[node] != -1:  # While not a leaf node
                feature_index = tree.feature[node]
                threshold = tree.threshold[node]
                if sample[feature_index] <= threshold:
                    node = tree.children_left[node]
                else:
                    node = tree.children_right[node]

            # Accumulate probabilities for the leaf
            probabilities.append(self._weighted_leaf_proba(tree, node))

        # Normalize to ensure probabilities sum to 1
        probabilities = np.array(probabilities)
        return probabilities / probabilities.sum(axis=1, keepdims=True)


##############
print(pd.Series(y_train).value_counts())
# # Train the model
model = WeightedLeafDecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = np.argmax(model.predict_proba(X_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Weighted : {accuracy:.4f}")
# print(classification_report(y_test, y_pred))

normal_model = DecisionTreeClassifier(random_state=42)
normal_model.fit(X_train, y_train)
y_pred = normal_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Regular DT: {accuracy:.4f}")
# print(classification_report(y_test, y_pred))

# # Train the model
model = DepthWeightedDecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = np.argmax(model.predict_proba(X_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Depth Weighted: {accuracy:.4f}")
# print(classification_report(y_test, y_pred))

soft = SoftSplitDecisionTreeClassifier(alpha=0.15, n_simulations=100, random_state=42)
soft.fit(X_train, y_train)
y_pred = np.argmax(soft.predict_proba(X_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Soft Decision: {accuracy:.4f}")
y_pred_train = np.argmax(soft.predict_proba(X_train), axis=1)
print(f"Accuracy Soft Decision Train: {accuracy_score(y_train, y_pred_train):.4f}")


# # RF
rf = RandomForestClassifier(random_state=42, n_estimators=25)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Random Forrest: {accuracy:.4f}")


adaptive_boosting = AdaptiveBoostingWeightedDecisionTreeClassifier(random_state=42)
adaptive_boosting.fit(X_train, y_train)
y_pred = np.argmax(adaptive_boosting.predict_proba(X_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy ADA: {accuracy:.4f}")

bayesian = BayesianWeightedDecisionTreeClassifier(random_state=42,prior_probs=priors_dict)
bayesian.fit(X_train, y_train)
y_pred = np.argmax(bayesian.predict_proba(X_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Baysian: {accuracy:.4f}")


kde = KDEWeightedDecisionTreeClassifier(random_state=42,bandwidth=0.5)
kde.fit(X_train, y_train)
y_pred = np.argmax(kde.predict_proba(X_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy KDE: {accuracy:.4f}")


entropy_model = EntropyWeightedDecisionTreeClassifier(random_state=42)
entropy_model.fit(X_train, y_train)
y_pred = np.argmax(entropy_model.predict_proba(X_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Entropy Weighted: {accuracy:.4f}")

class_weights = {0: 1.0, 1: 0.5, 2: 2.0}
class_specific = ClassSpecificWeightedDecisionTreeClassifier(random_state=42,class_weights=class_weights)
class_specific.fit(X_train, y_train)
y_pred = np.argmax(class_specific.predict_proba(X_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Class Specific: {accuracy:.4f}")

attention = LegalAttentionDecisionTreeClassifier(random_state=42,attention_temperature=0.75)
attention.fit(X_train, y_train)
y_pred = np.argmax(attention.predict_proba(X_test), axis=1)
y_pred_train = np.argmax(attention.predict_proba(X_train), axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Attention: {accuracy:.4f}")
print(f"Accuracy Attention Train: {accuracy_score(y_train, y_pred_train):.4f}")

entropy_regularized = EntropyRegularizedDecisionTreeClassifier(random_state=42)
entropy_regularized.fit(X_train, y_train)
y_pred = np.argmax(entropy_regularized.predict_proba(X_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Entropy Regularized: {accuracy:.4f}")
y_pred_train = np.argmax(entropy_regularized.predict_proba(X_train), axis=1)
print(f"Accuracy Entropy Regularized Train: {accuracy_score(y_train, y_pred_train):.4f}")
