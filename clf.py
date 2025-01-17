from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

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

class Classifier:
    def __init__(self, model_name='gmm', **kwargs):
        self.model_name = model_name
        if model_name == 'svm':
            kwargs.setdefault('nu', 0.2)
            self.model = OneClassSVM(**kwargs)
        elif model_name == 'gmm':
            self.model = GaussianMixture(**kwargs)
        elif model_name == 'kde':
            self.model = None
        else:
            raise ValueError("Unsupported model name. Choose from 'svm', 'gmm', or 'kde'.")
        
    def train(self, data):
        if self.model_name == 'svm':
            self.model.fit(data)
        elif self.model_name == 'gmm':
            self.model.fit(data)
            preds = self.model.score_samples(data)
            self.threshold = np.percentile(preds, 5)
        elif self.model_name == 'kde':
            self.model = gaussian_kde(data.T, bw_method='scott')
            preds = self.model.logpdf(data.T)
            self.threshold = np.percentile(preds, 5)

    def pred(self, data):
        if self.model_name == 'svm':
            preds = self.model.predict(data)
            return preds == 1
        elif self.model_name == 'gmm':
            preds = self.model.score_samples(data)
            return preds > self.threshold
        elif self.model_name == 'kde':
            preds = self.model.logpdf(data.T)
            return preds > self.threshold

    def __call__(self, data):
        return self.pred(data)
    
    def eval(self, id_data, ood_data):
        preds_real = self(id_data)
        preds_fake = self(ood_data)

        y = np.array([True] * len(preds_real) + [False] * len(preds_fake))
        y_pred = np.concatenate([preds_real, preds_fake])

        auroc = roc_auc_score(y, y_pred)
        fpr95 = calculate_fpr_at_tpr(y, y_pred, target_tpr=0.95)

        # Calculate AUPRC
        precision, recall, _ = precision_recall_curve(y, y_pred)
        auprc = average_precision_score(y, y_pred)

        # Calculate F1 score
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_score = np.max(f1_scores)
        
        return {'auroc': auroc,'fpr95': fpr95,'auprc': auprc,'f1_score': f1_score,}

# # Generate reference and test data
# ref_data = sample_gaussian(n=1000, mean=[0,0], std=[1,1])
# test_data = sample_gaussian(n=1000, mean=[100,100], std=[1,1])

# # Initialize and train the classifier
# clf = Classifier(model_name='gmm')
# clf.train(ref_data)

# # Predict on test data
# preds = clf(test_data)
# clf.eval(ref_data, test_data)

# def predmap(clf, xrange=(-10,10), yrange=(-10,10), n_samples=500, **kwargs):
#     xx, yy = np.meshgrid(np.linspace(*xrange, n_samples),
#                          np.linspace(*yrange, n_samples))
    
#     Z = clf(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     fig, ax = plt.subplots(**kwargs)
#     ax.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.2, colors=['red', 'green'])
#     return ax
