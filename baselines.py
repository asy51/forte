# Implements statistical baselines.

import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial import distance
from scipy.stats import zscore, ks_2samp, mannwhitneyu, entropy
from scipy.stats import wasserstein_distance
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

def calculate_mahalanobis_distance(X_id, X_ood):
    """
    Calculates the Mahalanobis distance between real and fake data.

    Args:
        X_id (numpy.ndarray): Real data samples.
        X_ood (numpy.ndarray): Fake data samples.

    Returns:
        numpy.ndarray: Mahalanobis distances for each fake data sample.
    """
    epsilon = 1e-5  # Small regularization factor, s.t. no divide by zero error.
    cov_matrix = np.cov(X_id.T)
    cov_matrix += np.eye(cov_matrix.shape[0]) * epsilon  # Regularization
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    mean_diffs = X_ood - np.mean(X_id, axis=0)
    return np.sqrt(np.diag(np.dot(np.dot(mean_diffs, inv_cov_matrix), mean_diffs.T)))

def compute_divergence(p, q, epsilon=1e-10):
    """
    Computes the KL divergence and JS divergence between two distributions with regularization.
    Args:
        p (numpy.ndarray): First distribution.
        q (numpy.ndarray): Second distribution.
        epsilon (float, optional): Small value for regularization. Defaults to 1e-10.
    Returns:
        tuple: A tuple containing the KL divergence and JS divergence.
    """
    p += epsilon
    q += epsilon
    p /= p.sum()
    q /= q.sum()
    kl_div = np.sum(p * np.log(p / q))
    js_div = (kl_div + np.sum(q * np.log(q / p))) / 2
    return kl_div, js_div

def bhattacharyya_distance(p, q, epsilon=1e-10):
    """
    Calculates the Bhattacharyya distance between two distributions.
    Args:
        p (numpy.ndarray): First distribution.
        q (numpy.ndarray): Second distribution.
        epsilon (float, optional): Small value for regularization. Defaults to 1e-10.
    Returns:
        float: Bhattacharyya distance between the distributions.
    """
    p = p + epsilon
    q = q + epsilon
    p /= p.sum()
    q /= q.sum()
    # Compute the Bhattacharyya distance
    return -np.log(np.sum(np.sqrt(p * q)))

def calculate_mutual_information(X_id, X_ood):
    """
    Calculates the mutual information between real and fake data.

    Args:
        X_id (numpy.ndarray): Real data samples.
        X_ood (numpy.ndarray): Fake data samples.

    Returns:
        float: Mutual information score.
    """
    est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
    X_id_disc = est.fit_transform(X_id.reshape(-1, 1)).ravel()
    X_ood_disc = est.fit_transform(X_ood.reshape(-1, 1)).ravel()
    return mutual_info_score(X_id_disc, X_ood_disc)

def perform_stats_tests(X_id, X_ood, label):
    """
    Performs various statistical tests between real and fake data and returns the results.

    Args:
        X_id (numpy.ndarray): Real data samples.
        X_ood (numpy.ndarray): Fake data samples.
        label (str): Label for the data being compared.

    Returns:
        dict: A dictionary containing the test results.
    """
    results = {}

    # Subset data to match the smaller dataset
    n_samples = min(len(X_id), len(X_ood))
    X_id_subset = X_id[:n_samples]
    X_ood_subset = X_ood[:n_samples]

    # Z-Score
    z_scores = zscore(X_id_subset - X_ood_subset)
    results[f"Z-Score"] = np.mean(z_scores)

    # K-S Test
    ks_stat, ks_p = ks_2samp(X_id.ravel(), X_ood.ravel())
    results[f"K-S Test"] = {"stat": ks_stat, "p-value": ks_p}

    # Mann-Whitney U Test
    mann_stat, mann_p = mannwhitneyu(X_id.ravel(), X_ood.ravel(), alternative='two-sided')
    results[f"Mann-Whitney U Test"] = {"stat": mann_stat, "p-value": mann_p}

    # Local Outlier Factor
    lof = LocalOutlierFactor(novelty=True)
    lof.fit(X_id)
    lof_scores = lof.decision_function(X_ood)
    results[f"Local Outlier Factor"] = np.mean(lof_scores)

    # Isolation Forest
    iso_forest = IsolationForest(random_state=42)
    iso_forest.fit(X_id)
    iso_scores = iso_forest.decision_function(X_ood)
    results[f"Isolation Forest"] = np.mean(iso_scores)

    p = X_id.mean(axis=0)
    q = X_ood.mean(axis=0)
    kl_div, js_div = compute_divergence(p, q)

    # Jensen-Shannon Divergence
    results[f"Jensen-Shannon Divergence"] = js_div

    # KL Divergence
    results[f"KL Divergence"] = kl_div

    # Wasserstein Distance
    wasserstein_result = wasserstein_distance(X_id.ravel(), X_ood.ravel())
    results[f"Wasserstein Distance"] = wasserstein_result

    # Mahalanobis Distance
    mdist = calculate_mahalanobis_distance(X_id_subset, X_ood_subset)
    results[f"Mahalanobis Distance"] = np.mean(mdist)

    # Bhattacharyya Distance
    b_distance = bhattacharyya_distance(p, q)
    results[f"Bhattacharyya Distance"] = b_distance

    # Mutual Information
    mi = calculate_mutual_information(X_id_subset.ravel(), X_ood_subset.ravel())
    results[f"Mutual Information"] = mi

    return results