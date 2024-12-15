import numpy as np
from prdc import compute_nearest_neighbour_distances, compute_pairwise_distance

def compute_prdc_per_point(real_features, fake_features, nearest_k, realism=False):
    """
    Computes precision, recall, density, coverage, and optionally realism per fake data point.

    Args:
        real_features (np.ndarray): Features of real data points, shape (N, feature_dim).
        fake_features (np.ndarray): Features of fake data points, shape (M, feature_dim).
        nearest_k (int): Number of nearest neighbors to consider.
        realism (bool): Whether to compute realism scores.

    Returns:
        dict: Contains arrays for precision, recall, density, coverage, and optionally realism.
              Each array has shape (M,) where M is the number of fake samples.
    """
    num_real = real_features.shape[0]
    num_fake = fake_features.shape[0]

    real_distances = compute_nearest_neighbour_distances(real_features, nearest_k)
    fake_distances = compute_nearest_neighbour_distances(fake_features, nearest_k)
    distance_matrix = compute_pairwise_distance(real_features, fake_features)

    precision = compute_precision(distance_matrix, real_distances)
    recall = compute_recall(distance_matrix, fake_distances, num_real)
    density = compute_density(distance_matrix, real_distances, nearest_k, num_real)
    coverage = compute_coverage(distance_matrix, fake_distances)

    result = {
        "precision": precision,
        "recall": recall,
        "density": density,
        "coverage": coverage
    }

    if realism:
        result["realism"] = compute_realism(distance_matrix, real_distances)

    return result

def compute_precision(distance_matrix, real_distances):
    """Compute precision for each fake point."""
    return (distance_matrix < real_distances[:, np.newaxis]).any(axis=0).astype(float)

def compute_recall(distance_matrix, fake_distances, num_real):
    """Compute recall for each fake point."""
    return (distance_matrix < fake_distances).sum(axis=0).astype(float) / num_real

def compute_density(distance_matrix, real_distances, nearest_k, num_real):
    """Compute density for each fake point."""
    return (distance_matrix < real_distances[:, np.newaxis]).sum(axis=0).astype(float) / (nearest_k * num_real)

def compute_coverage(distance_matrix, fake_distances):
    """Compute coverage for each fake point."""
    return (distance_matrix.min(axis=0) < fake_distances).astype(float)

def compute_realism(distance_matrix, real_distances):
    """Compute realism for each fake point."""
    median_distance = np.median(real_distances)
    mask = real_distances < median_distance
    real_distances_masked = real_distances[mask][:, np.newaxis]
    distance_matrix_masked = distance_matrix[mask]
    realism = (real_distances_masked / distance_matrix_masked).max(axis=0)
    return np.squeeze(realism)