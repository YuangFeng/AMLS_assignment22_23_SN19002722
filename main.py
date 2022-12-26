
from utils.train import train_A1, train_A2, train_B1, train_B2
        
if __name__ == '__main__':    
   train_A1()
   """
   ##### A1 results(40 epochs)#######
   1.test acc:0.953
   2.test f1:0.9533267130089375
   3.confusion matrix: tn [473], fp [27], fn [20], tp [480]
   """
   #train_A2()
   """
      ##### A2 results #######
      Result from grid search:
         C=10000, kernel=’rbf’
      A2 testing results:
      1.test acc:0.9030927835051547
      2.test f1:0.906187624750499
      3.confusion matrix: tn [422], fp [50], fn [44], tp [454]
   """
   # train_B1()
   """
      ####### B1 results ##########
      Result from grid search:
         best score: 0.9999
         best parameters: {'n_estimators': 150}
      1.test acc:1
      2.test f1:[1.  1.  1.       1.       1.      ]
   """
   #train_B2()
   """
      ##### B2 results #########
      training set:
         total image without glasses: 8142 total:10000 rate:0.8142
      testing set:
         total image without glasses: 2033 total:2500 rate: 0.8132
      Results from grid search:
         best score: 0.9853843044450781
         best parameters: {'n_estimators': 100}
      1.test acc:0.9881947860304968
      2.test f1:[0.98636927 0.98473282 0.98734177 0.98424242 0.9987163 ]
   """