from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score,GridSearchCV
class Model_B2:
    def __init__(self) -> None:
        self.model = RandomForestClassifier(n_estimators=50, n_jobs=-1)
        self.parameters = {'n_estimators':(50,100,150,200)}
        self.clf = GridSearchCV(RandomForestClassifier(), self.parameters, scoring='f1', n_jobs=-1)
        
    def train(self, x, y):
        """
        Train the model
        Input parameters:
            x: pre-processed images in training set
            y: labels of images in training set
        Returns:
            None
        """
        # scores = cross_val_score(self.model, x, y, cv=10)
        # print('K-fold scores:', scores)
        self.model = self.clf.fit(x,y)
        print('best score:',self.clf.best_score_)
        print('best parameters:', self.clf.best_params_)
    
    def test(self, x, y):
        """
        Test the trained model
        Input parameters:
            x: images in testing set
            y: labels of images in testing set
        Returns:
            acc: accuracy of model 
            p_class: precision of model
            r_class: Recall of model
            f_class: F1 score of model
        """
        pred = self.model.predict(x)
        acc = accuracy_score(y, pred)
        p_class, r_class, f_class, support_micro = precision_recall_fscore_support(y, pred)
        return  acc, p_class, r_class, f_class