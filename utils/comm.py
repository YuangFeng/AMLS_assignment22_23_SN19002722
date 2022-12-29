import matplotlib.pyplot as plt
from sklearn.metrics import auc
import numpy as np
import seaborn as sns


def plot_learning_curve(train_sizes, train_scores, test_scores, save_file):
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='train score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color = 'b', label = 'test score')
    plt.xlabel('Traning examples')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.savefig(save_file)
    plt.close()

def plot_cm(cm, save_file):
    fig = sns.heatmap(cm, annot=True, fmt='d',cmap = 'Blues')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    heatmap = fig.get_figure()
    heatmap.savefig(save_file, dpi = 400)
    plt.close()
        

def plot_roc(roc, save_file):
    """
    Plot and save the ROC curve
    Input parameters: 
        roc: fpr, tpr, thersholds
    Returns:
        None
    """
    fpr, tpr, thersholds = roc
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)       
    plt.xlim([-0.05, 1.05])  # Set the limit of x label and y label to observe the graph properly
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_file)
    plt.close()