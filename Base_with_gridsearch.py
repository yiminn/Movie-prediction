import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score,cross_val_predict,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import  MultinomialNB
from sklearn.metrics import recall_score, precision_score,f1_score
import pickle



def obtaindata(data):
    X_train = pd.read_csv('/Users/DYM/PycharmProjects/Movie-prediction/ProcessedData/'+data+'.csv')
    X_train = X_train.fillna(0)
    y_train = pd.read_csv('Dev_Set/dev_set_groundtruth_and_trailers.csv')['goodforairplane']
    return (X_train,y_train)


def creat_name(data_type,model):
    model_name1 = str(model).split("'")[-2]
    model_name = model_name1.split('.')[-1]
    combining = '_'.join([data_type,model_name])
    return combining, model_name

def gridcv(model):
    if model == KNeighborsClassifier:
        param_grid={'n_neighbors':[4, 5, 6, 7, 8, 9],"weights":['uniform','distance']}
    elif model == MultinomialNB:
        param_grid={'alpha':[0.001,0.1,1,10]}
    elif model == DecisionTreeClassifier:
        param_grid={"max_depth":np.arange(6, 15)}
    elif model == LogisticRegression:
        param_grid={"C":[0.001,0.1,1,10,100]}
    elif model == SVC:
        param_grid={"C":[0.01,0.1, 1, 10,100], "gamma": [1, 0.1, 0.01],'probability':[True]}
    elif model ==BaggingClassifier:
        param_grid = {"n_estimators":[50,100,200],"max_samples":[0.7,0.8,0.9],'max_features':[0.7,0.8,0.9]}
    elif model == RandomForestClassifier:
        param_grid = {"n_estimators": [50, 100,200]}
    else:
        param_grid = {"n_estimators":[50,100,200],"learning_rate":[0.01,0.1,0.001]}
    return param_grid




def modelling(data,clf):
    X_train, y_train = obtaindata(data)
    param = gridcv(clf)
    model =clf()
    grid = GridSearchCV(model, param_grid=param, cv=7, scoring='accuracy')
    grid.fit(X_train, y_train)
    best_param=grid.best_params_
    pre_score = round(grid.best_score_,3)
    # prediction = grid.predict(X_train)
    #f1 = f1_score(y_train,prediction)
    #rec_score = recall_score(y_train,prediction)
    _,name = creat_name(data, clf)
    print(name)
    print('precision: %.3f ' %(pre_score))
    print('\n-------------------------------\n')
    return pre_score,best_param



if __name__ == '__main__':

    base_model = [KNeighborsClassifier, MultinomialNB, DecisionTreeClassifier, LogisticRegression,
                  SVC, BaggingClassifier, RandomForestClassifier, AdaBoostClassifier,
                  GradientBoostingClassifier]
    typeDataset = ( 'TextData','V3Data','AudioData', 'VisualData','ProcessedMeta')
    # for data in typeDataset:
    #    result(data=data)

    # stack_matrix ={}
    metrics = {}
    for data in typeDataset:
        detail_set = []
        print('====================',data,'=====================================')
        for model in base_model:
            try:
                detail = []
                combining, model_name = creat_name(data, model)
                pre_score , best_param = modelling(data,model)
                if pre_score>0.5:#and pre_score != 0.708 and pre_score != 0.707:
                    detail = [model_name,best_param,pre_score]
                    detail_set.append(detail)
            except ValueError:
                print(combining, ': has error\n-------------------------------\n')
                pass
        metrics[data]= detail_set
    print(metrics)
    file1=open('/Users/DYM/PycharmProjects/Movie-prediction/Data_Record/detail_accuracy.txt','wb')
    pickle.dump(metrics,file1)
    file1.close()

