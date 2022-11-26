
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_curve, confusion_matrix, auc
from matplotlib import pyplot as plt

class Model_A2:
    def __init__(self, search=False) -> None:
        self.search = search
        self.model = svm.SVC(C=10000, gamma=2.6826957952797274e-06, probability=True)
        # self.model = RandomForestClassifier(n_estimators=10, n_jobs=-1)
        if self.search:
            self.parameters = {'kernel':('rbf','linear'),'C':(1,2,3)}
            self.clf = GridSearchCV(svm.SVC(probability=True), self.parameters, scoring='f1', n_jobs=-1)
        
    
    def train(self, x, y):
        if self.search:
            self.model = self.clf.fit(x,y)
            print('best score:',self.clf.best_score_)
            print('best parameters:', self.clf.best_params_)
        else:
            self.model = self.model.fit(x,y)
        return self.test(x,y)
    
    def test(self, x, y):
        pred = self.model.predict(x)
        pred_score = self.model.predict_proba(x)[:,1]
        acc = accuracy_score(y, pred)
        f1 = f1_score(y, pred)
        roc = roc_curve(y, pred_score, pos_label=1) #fpr, tpr, thersholds
        cm = confusion_matrix(y, pred).ravel()
        self.plot_roc(roc)
        return acc, f1, roc, cm
    
    def plot_roc(self, roc):
        fpr, tpr, thersholds = roc
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
        
        plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig('A2_ROC.jpg')
