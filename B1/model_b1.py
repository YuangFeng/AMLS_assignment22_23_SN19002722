from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score
class Model_B1:
    def __init__(self) -> None:
        self.model = RandomForestClassifier(n_estimators=50, n_jobs=-1)
    def train(self, x, y):
        # scores = cross_val_score(self.model, x, y, cv=10)
        # print('K-fold scores:', scores)
        self.model = self.model.fit(x,y)
    
    def test(self, x, y):
        pred = self.model.predict(x)
        acc = accuracy_score(y, pred)
        p_class, r_class, f_class, support_micro = precision_recall_fscore_support(y, pred)
        return  acc, p_class, r_class, f_class
