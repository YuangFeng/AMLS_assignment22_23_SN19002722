
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_curve, confusion_matrix, auc
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score

class Model_A2:
    """
    The model used in task A2 is svm. one can choose to use gridsearch to find the hyperparameters of model.
    parameters: 
        search: bool, whether to use grid search, default:False
    """
    def __init__(self, search=False) -> None:
        self.search = search
        self.model = svm.SVC(C=10000, gamma=2.6826957952797274e-06, probability=True)
        # self.model = RandomForestClassifier(n_estimators=10, n_jobs=-1)
        if self.search:
            self.parameters = {'kernel':('rbf','linear'),'C':(1,2,3)}
            self.clf = GridSearchCV(svm.SVC(probability=True), self.parameters, scoring='f1', n_jobs=-1)
        
    
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
            # scores = cross_val_score(self.model, x, y, cv = 10, scoring = 'f1')
            # print('k-fold scores:', scores)
            self.model = self.model.fit(x, y)
            
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
        cm = confusion_matrix(y, pred).ravel()
        self.plot_roc(roc)
        return acc, f1, roc, cm
    
    def plot_roc(self, roc):
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
        plt.savefig('A2_ROC.jpg')
