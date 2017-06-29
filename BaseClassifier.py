import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import  MultinomialNB

def result(data):
    base_model = (KNeighborsClassifier,DecisionTreeClassifier,LogisticRegression,
             SVC,BaggingClassifier,RandomForestClassifier,AdaBoostClassifier,
             GradientBoostingClassifier,MultinomialNB)
    df = pd.read_csv('/Users/DYM/PycharmProjects/MasterFYP/ProcessedData/'+data+'.csv')
    df = df.fillna(0)
    y_train = df.goodforairplane
    X_train = df.drop(['goodforairplane'],axis=1)
    for base_clf in base_model:
        clf = base_clf()
        scores1 = cross_val_score(clf, X_train, y_train, cv=10,scoring='f1')
        scores2 = cross_val_score(clf, X_train, y_train, cv=10,scoring='precision')
        scores3 = cross_val_score(clf, X_train, y_train, cv=10,scoring='recall')
        f1_score = scores1.mean()
        precision_score = scores2.mean()
        recall_score = scores3.mean()
        print(str(base_clf))
        print('f1: %.3f \nprecesion: %f \nrecall: %f' %(f1_score, precision_score, recall_score))

if __name__ == '__main__':
    typeDataset = ('AudioData', 'TextualData', 'V3Data', 'VisualData')
    for data in typeDataset:
       result(data=data)
    #result('AudioData')