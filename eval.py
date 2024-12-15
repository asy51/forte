import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from scipy.stats import gaussian_kde

def calculate_fpr_at_tpr(y_true, y_scores, target_tpr=0.95):
    """
    Calculate the False Positive Rate (FPR) at a given True Positive Rate (TPR) threshold.
    
    Args:
    y_true (array-like): True binary labels.
    y_scores (array-like): Target scores (can be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions).
    target_tpr (float): The desired TPR threshold (default: 0.95).
    
    Returns:
    float: The FPR at the given TPR threshold.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return fpr[next(i for i, x in enumerate(tpr) if x >= target_tpr)]

def evaluate_ocsvm(X_train_id, X_test_id, X_test_ood):
    """
    Evaluate One-Class SVM for OOD detection.
    
    Args:
    X_train_id (array-like): Training data from the real distribution.
    X_test_id (array-like): Held-out data from the real distribution.
    X_test_ood (array-like): Held-out data from the fake (OOD) distribution.
    
    Returns:
    tuple: AUROC, FPR@95TPR, AUPRC, and F1 score.
    """
    held_out_data = np.concatenate([X_test_id, X_test_ood], axis=0)
    held_out_labels = np.array([1] * len(X_test_id) + [-1] * len(X_test_ood))

    best_score = -float('inf')
    best_nu = None

    # Hyperparameter tuning (ONLY ON TRAIN SET)
    for nu_candidate in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        oc_svm = OneClassSVM(kernel='rbf', gamma='auto', nu=nu_candidate)
        oc_svm.fit(X_train_id)
        inliers_ratio = (oc_svm.predict(X_train_id) == 1).sum() / len(X_train_id)

        if inliers_ratio > best_score:
            best_score = inliers_ratio
            best_nu = nu_candidate

    # Train final model with best hyperparameters
    oc_svm = OneClassSVM(kernel='rbf', gamma='auto', nu=best_nu)
    oc_svm.fit(X_train_id)

    held_out_preds = oc_svm.decision_function(held_out_data)
    held_out_auroc = roc_auc_score(held_out_labels, held_out_preds)
    held_out_fpr95 = calculate_fpr_at_tpr(held_out_labels, held_out_preds, target_tpr=0.95)
    
    # Calculate AUPRC
    precision, recall, _ = precision_recall_curve(held_out_labels, held_out_preds)
    auprc = average_precision_score(held_out_labels, held_out_preds)
    
    # Calculate F1 score
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_score = np.max(f1_scores)

    return held_out_auroc, held_out_fpr95, auprc, f1_score

def evaluate_kde(X_train_id, X_test_id, X_test_ood):
    """
    Evaluate Kernel Density Estimation for OOD detection.
    
    Args:
    X_train_id (array-like): Training data from the real distribution.
    X_test_id (array-like): Held-out data from the real distribution.
    X_test_ood (array-like): Held-out data from the fake (OOD) distribution.
    
    Returns:
    tuple: AUROC, FPR@95TPR, AUPRC, and F1 score.
    """
    try: 
        best_bw = None
        best_kde_score = -float('inf')
        
        # Hyperparameter tuning (ONLY ON TRAIN SET)
        for bw in ['scott', 'silverman']:
            kde = gaussian_kde(X_train_id.T, bw_method=bw)
            score = kde.logpdf(X_train_id.T).mean()
            if score > best_kde_score:
                best_kde_score = score
                best_bw = bw

        kde = gaussian_kde(X_train_id.T, bw_method=best_bw)
        preds_real = kde.logpdf(X_test_id.T)
        preds_fake = kde.logpdf(X_test_ood.T)
        
        y_true = np.array([1] * len(preds_real) + [-1] * len(preds_fake))
        y_scores = np.concatenate([preds_real, preds_fake])
        
        auroc = roc_auc_score(y_true, y_scores)
        fpr95 = calculate_fpr_at_tpr(y_true, y_scores, target_tpr=0.95)
        
        # Calculate AUPRC
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        auprc = average_precision_score(y_true, y_scores)
        
        # Calculate F1 score
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_score = np.max(f1_scores)
    
        return auroc, fpr95, auprc, f1_score
    except:
        return 0, 0, 0, 0

def evaluate_gmm(X_train_id, X_test_id, X_test_ood):
    """
    Evaluate Gaussian Mixture Model for OOD detection.
    
    Args:
    X_train_id (array-like): Training data from the real distribution.
    X_test_id (array-like): Held-out data from the real distribution.
    X_test_ood (array-like): Held-out data from the fake (OOD) distribution.
    
    Returns:
    tuple: AUROC, FPR@95TPR, AUPRC, and F1 score.
    """
    best_n_components = None
    best_gmm_score = -float('inf')
    
    # Hyperparameter tuning (ONLY ON TRAIN SET)
    for n_components in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        gmm = GaussianMixture(n_components=n_components)
        gmm.fit(X_train_id)
        score = gmm.score(X_train_id)
        if score > best_gmm_score:
            best_gmm_score = score
            best_n_components = n_components

    gmm = GaussianMixture(n_components=best_n_components)
    gmm.fit(X_train_id)
    
    preds_real = gmm.score_samples(X_test_id)
    preds_fake = gmm.score_samples(X_test_ood)
    
    y_true = np.array([1] * len(preds_real) + [-1] * len(preds_fake))
    y_scores = np.concatenate([preds_real, preds_fake])
    
    auroc = roc_auc_score(y_true, y_scores)
    fpr95 = calculate_fpr_at_tpr(y_true, y_scores, target_tpr=0.95)
    
    # Calculate AUPRC
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)
    
    # Calculate F1 score
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_score = np.max(f1_scores)
    
    return auroc, fpr95, auprc, f1_score

def run_evaluations(X_train_id, X_test_id, X_test_ood):
    """
    Run evaluations for all OOD detection methods.
    
    Args:
    X_train_id (array-like): Training data from the real distribution.
    X_test_id (array-like): Held-out data from the real distribution.
    X_test_ood (array-like): Held-out data from the fake (OOD) distribution.
    
    Returns:
    dict: Results for each method, including AUROC, FPR@95TPR, AUPRC, and F1 score.
    """
    # Evaluate using Kernel Density Estimation
    auroc_kde, fpr95_kde, auprc_kde, f1_kde = evaluate_kde(X_train_id, X_test_id, X_test_ood)
    
    # Evaluate using Gaussian Mixture Models
    auroc_gmm, fpr95_gmm, auprc_gmm, f1_gmm = evaluate_gmm(X_train_id, X_test_id, X_test_ood)
    
    # Evaluate using One-Class SVM
    auroc_ocsvm, fpr95_ocsvm, auprc_ocsvm, f1_ocsvm = evaluate_ocsvm(X_train_id, X_test_id, X_test_ood)
    
    # Aggregate results into a dictionary
    results = {
        "KDE": {"AUROC": auroc_kde, "FPR@95TPR": fpr95_kde, "AUPRC": auprc_kde, "F1": f1_kde},
        "GMM": {"AUROC": auroc_gmm, "FPR@95TPR": fpr95_gmm, "AUPRC": auprc_gmm, "F1": f1_gmm},
        "OCSVM": {"AUROC": auroc_ocsvm, "FPR@95TPR": fpr95_ocsvm, "AUPRC": auprc_ocsvm, "F1": f1_ocsvm}
    }
    
    return results