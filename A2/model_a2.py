
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_curve, confusion_matrix, auc
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score, learning_curve
import seaborn as sns
import numpy as np
from utils.comm import plot_learning_curve, plot_cm, plot_roc



class Model_A2:
    """
    The model used in task A2 is svm. one can choose to use gridsearch to find the hyperparameters of model.
    parameters: 
        search: bool, whether to use grid search, default:False
    """
    def __init__(self, search=True):
        self.search = search
        self.model = svm.SVC(kernel = 'rbf', C=10000, probability=True)
        # self.model = RandomForestClassifier(n_estimators=10, n_jobs=-1)
        if self.search:
            self.parameters = {'kernel':('rbf','linear'),'C':(10000,20000,30000)}
            self.clf = GridSearchCV(svm.SVC(probability=True), self.parameters, scoring='f1', n_jobs=-1, cv = 5)
        
    
    def train(self, x, y):
        """
        Trian model according to given training data
        Input parameters:
            x:Tensor, features of images
            y:Tensor, labels of images
        
        Returns:
            None
        """
        if self.search:
            self.model = self.clf.fit(x,y)
            print('best score:',self.clf.best_score_)
            print('best parameters:', self.clf.best_params_)
        else:
            self.model = self.model.fit(x, y)
            scores = cross_val_score(self.model, x, y, cv = 5, scoring = 'f1')
            print('k-fold scores:', scores)
            
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, x, y, cv=3, n_jobs = -1, train_sizes=np.linspace(.1, 1.0, 5), scoring='f1'
        )
        plot_learning_curve(train_sizes, train_scores, test_scores, 'A2_learning_curve')
        
            
    def test(self, x, y):
        """
        Test model with given test data
        Input parameters:
            x:Tensor, features of images
            y:Tensor, labels of images
        
        Returns:
            acc: accuracy of model
            f1: F1-score of model
            roc: roc curve of model, (fpr, tpr, thersholds)
            cm: confusion matrix of model
        """
        pred = self.model.predict(x)
        pred_score = self.model.predict_proba(x)[:,1]
        acc = accuracy_score(y, pred)
        f1 = f1_score(y, pred)
        roc = roc_curve(y, pred_score, pos_label=1) #fpr, tpr, thersholds
        cm = confusion_matrix(y, pred)
        plot_roc(roc, 'A2_ROC.jpg')
        plot_cm(cm, 'A2_heatmap.png')
        return acc, f1, roc, cm
    

        

