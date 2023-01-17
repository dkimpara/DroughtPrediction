import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import re
import warnings
import time    
import joblib


from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class OrdinalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, loss=None):
        self.base_estimator = base_estimator
        self.loss = loss #callable

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        
        self.classes_ = np.sort(unique_labels(y))
        self.X_ = X
        self.y_ = y
        
        self.n_features_in_ = X.shape[1]
        
        self.estimators_ = {}

        for i in range(self.classes_.shape[0]-1):
            # for each k - 1 ordinal value we fit a binary classification problem
            binary_y = (y > self.classes_[i]).astype(np.uint8)
            clf = clone(self.base_estimator)
            clf.fit(X, binary_y)
            self.estimators_[self.classes_[i]] = clf
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        clfs_predict = {k: self.estimators_[k].predict_proba(X) for k in self.estimators_}
        predicted = []
        
        if len(self.classes_) == 1:
            return np.ones((X.shape[0],1))
        
        # V1 = 1 - Pr(y > V1)
        predicted.append(1 - clfs_predict[self.classes_[0]][:,1])
        
        for i in range(1, len(self.classes_) - 1): #sorted classes
            # Vi = Pr(y > Vi-1) - Pr(y > Vi)
             predicted.append(clfs_predict[self.classes_[i-1]][:,1] - 
                              clfs_predict[self.classes_[i]][:,1])
        
        # Vk = Pr(y > Vk-1)
        predicted.append(clfs_predict[self.classes_[-2]][:,1])
        return np.vstack(predicted).T

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y, sample_weight=None):
        _, indexed_y = np.unique(y, return_inverse=True)
        return self.loss(indexed_y, self.predict(X), sample_weight=sample_weight)


def save_model(model, name):
#save your model or results
    joblib.dump(model, name + '.pkl')

#load your model for further usage

def load_model(name):
    return joblib.load(name)
    
def timer(tic):
    toc = time.perf_counter()
    duration = toc - tic
    minutes = int(duration/60)
    print(f"Elapsed time: {str(minutes) + ' minutes, ' if minutes else ''}{int(duration % 60)} seconds")

def round_and_intify(y):
    return np.clip(np.squeeze(y).round().astype('int32'), 0, 5)

def bold(string):
    return f'\033[1m{string}\033[0m'

def plot_confusion_matrix(matrix, labels):
    plt.figure(1, figsize=(8.7, 7))
    ax = sns.heatmap(matrix, annot=True, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticklabels(labels)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_yticklabels(labels, rotation=90, va="center")
    ax.set_xlabel("Predicted Label", labelpad=18, fontdict=dict(weight='bold'))
    ax.set_ylabel("True Label", labelpad=15, fontdict=dict(weight='bold'))
    colorax = plt.gcf().axes[-1]
    colorax.tick_params(length=0)
    plt.show()
    
def plot_confusion_matrices(matrices, labels, macro_f1s, maes):
    fig, axes = plt.subplots(1, 6, sharex=True, sharey=True, figsize=(20,5.6))
    for i, (matrix, macro_f1, mae) in enumerate(zip(matrices, macro_f1s, maes)):
        sns.heatmap(matrix, ax=axes[i], annot=True, cmap="YlGnBu", vmin=0, vmax=1, cbar=False)
        axes[i].set_xticklabels(labels)
        axes[i].xaxis.tick_top()
        axes[i].xaxis.set_label_position('top') 
        axes[i].tick_params(axis='both', which='both', length=0)
        axes[i].set_yticklabels(labels, rotation=90, va="center")
        axes[i].text(3, 7,f'MAE: {mae}\nMacro F1: {macro_f1}', fontsize=14, ha='center')
        axes[i].set_xlabel(f"Predicted Label\nWeek {i+1}", labelpad=15, fontdict=dict(weight='bold'), fontsize=14)
        axes[0].set_ylabel("True Label", labelpad=12, fontdict=dict(weight='bold'), fontsize=14)
    fig.suptitle(f'avg. MAE: {np.mean([float(x) for x in maes]):.3f}\n' \
                 f'avg. Macro F1: {np.mean([float(x) for x in macro_f1s]):.3f}', 
                 fontsize=14,
                 fontweight='bold',
                 y=0.02)
    fig.tight_layout()
    plt.show()
    
def summarize(y_true, y_pred):
    weeks_true = np.split(y_true, 6, 1)
    weeks_pred = np.split(y_pred, 6, 1)
    matrices = []
    macro_f1s = []
    maes = []
    for y_true, y_pred in zip(weeks_true, weeks_pred):
        y_true = round_and_intify(y_true.flatten())
        y_pred = round_and_intify(y_pred.flatten())
        matrix = metrics.confusion_matrix(y_true, y_pred, normalize='true')
        matrices += [matrix]
        report = metrics.classification_report(y_true, y_pred, digits=3)
        r = re.compile('(?<=macro avg\s{6}\d.\d{3}\s{5}\d.\d{3}\s{5})\d.\d{3}')
        macro_f1s += [r.search(report).group(0)]
        maes += [f"{np.mean(abs(y_true - y_pred)):.3f}"]
    warnings.filterwarnings('ignore')
    labels = ["None", "D0", "D1", "D2", "D3", "D4"]
    plot_confusion_matrices(matrices, labels, macro_f1s, maes)
    
def macro_f1(y_true, y_pred):
    weeks_true = np.split(y_true, 6, 1)
    weeks_pred = np.split(y_pred, 6, 1)
    macro_f1s = []
    for y_true, y_pred in zip(weeks_true, weeks_pred):
        y_true = round_and_intify(y_true.flatten())
        y_pred = round_and_intify(y_pred.flatten())
        macro_f1s.append(metrics.f1_score(y_true, y_pred, average='macro'))
    mf1 = np.mean([float(x) for x in macro_f1s])
    return mf1

def mae(y_true, y_pred):
    target = round_and_intify(y_true)
    pred = round_and_intify(y_pred)
    return metrics.mean_absolute_error(target, pred)