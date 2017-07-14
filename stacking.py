import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score,cross_val_predict,train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import  MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import recall_score, precision_score,f1_score
from xgboost.sklearn import XGBClassifier
import  xgboost
import sklearn
import pickle

typeDataset = ('TextData','AudioData', 'V3Data','VisualData','ProcessedMeta')


def loaddetail():
    f1 = open('/Users/DYM/PycharmProjects/Movie-prediction/Data_Record/type_model2.txt','rb')
    detail = pickle.load(f1)
    f1.close()
    print(detail)
    return detail

def obtaintrain(data):
    X_train = pd.read_csv('/Users/DYM/PycharmProjects/Movie-prediction/ProcessedData/'+data+'.csv')
    X_train = X_train.fillna(0)
    y_train = pd.read_csv('Dev_Set/dev_set_groundtruth_and_trailers.csv')['goodforairplane']
    return (X_train,y_train)

def obtaintest(data):
    X_test = pd.read_csv('/Users/DYM/PycharmProjects/Movie-prediction/ProcessedData/test_set/'+data+'.csv')
    X_test = X_test.fillna(0)
    return (X_test)

def obtainmodel(clf_str):
    if clf_str == 'KNeighborsClassifier':
        clf = getattr(sklearn.neighbors,clf_str)
    elif clf_str == 'DecisionTreeClassifier':
        clf = getattr(sklearn.tree,clf_str)
    elif clf_str == 'LogisticRegression':
        clf =getattr(sklearn.linear_model,clf_str)
    elif clf_str == "XGBClassifier":
        clf = getattr(xgboost.sklearn,clf_str)
    elif clf_str == 'SVC':
        clf = getattr(sklearn.svm,clf_str)
    elif clf_str == 'MultinomialNB':
        clf = getattr(sklearn.naive_bayes, clf_str)
    elif clf_str == 'BaggingClassifier' or 'RandomForestClassifier' or 'AdaBoostClassifier':
        clf = getattr(sklearn.ensemble, clf_str)
    return clf

def modelling(data,clf_str,param,method):
    X_train, y_train = obtaintrain(data)
    X_test = obtaintest(data)
    clf = obtainmodel(clf_str)
    model = clf(**param)
    model.fit(X_train, y_train)
    if method == 'predict_proba':
        y_test = model.predict_proba(X_test)[:,1]
    else:
        y_test = model.predict(X_test)
    return y_test

def gridcv(model):
    if model == KNeighborsClassifier:
        param_grid={'n_neighbors':[4, 5, 6, 7, 8, 9],"weights":['uniform','distance']}
    elif model == MultinomialNB:
        param_grid={'alpha':[0.001,0.1,1,10]}
    elif model == XGBClassifier:
        param_grid = {
        'max_depth': [3, 4, 5, 6, 7, 8, 9],
        'min_child_weight': [5, 6, 7, 8, 9],
        'gamma': [i / 10.0 for i in range(0, 1)],
        'subsample': [i / 10.0 for i in range(3, 4, 6)],
        'colsample_bytree': [i / 10.0 for i in range(5, 6)],
        'reg_alpha': [1e-5, 1e-2, 0.1, 1],
        'reg_lambda': [1e-5, 1e-2, 0.1, 1]
        }
    elif model == DecisionTreeClassifier:
        param_grid={"max_depth":np.arange(6, 15)}
    elif model == LogisticRegression:
        param_grid={"C":[0.001,0.1,1,10,100]}
    elif model == SVC:
        param_grid={"C":[0.01,0.1, 1, 10,100,1000], "gamma": [1, 0.1, 0.01,0.001],'probability':[True]}
    elif model ==BaggingClassifier:
        param_grid = {"n_estimators":[50,100,200],"max_samples":[0.7,0.8,0.9],'max_features':[0.7,0.8,0.9]}
    elif model == RandomForestClassifier:
        param_grid = {"n_estimators": [50, 100,200]}
    else:
        param_grid = {"n_estimators":[100,200,300],"learning_rate":[0.01,0.1,0.001]}
    return param_grid

def meta_level_train(detail,method):
    stack_matrix_train={}
    for data in typeDataset:
        print('=========================',"training data of ",data,' is processing==============')
        X_train, y_train = obtaintrain(data)
        subdtail = detail[data]
        for i in range(len(subdtail)):
            final_detail = subdtail[i]
            print('training data of ',final_detail[0],' is processing')
            clf = obtainmodel(final_detail[0])
            model = clf(**final_detail[1])
            if method == 'predict_proba':
                proba = cross_val_predict(model, X_train, y_train, cv=10,method=method)[:,1]
            else:
                proba = cross_val_predict(model, X_train, y_train, cv=10)
            stack_matrix_train[data+'_'+final_detail[0]] = proba
    df = pd.DataFrame(stack_matrix_train)
    df.to_csv('/Users/DYM/PycharmProjects/Movie-prediction/ProcessedData/base_result/StackingDataTrain_orig2.csv',index=False)



def create_stack_test(detail,method):
    stack_matrix_test={}
    for data in typeDataset:
        print('==============================',data, 'is processing===============')
        subdtail = detail[data]
        for i in range(len(subdtail)):
            final_detail = subdtail[i]
            print(final_detail[0],'is processing')
            y_test = modelling(data,final_detail[0],final_detail[1],method=method)
            stack_matrix_test[data+'_'+final_detail[0]] = y_test
    df = pd.DataFrame(stack_matrix_test)
    df.to_csv('/Users/DYM/PycharmProjects/Movie-prediction/ProcessedData/test_set/base_result/StackingDataTest_orig2.csv',index=False)
    return df


def meta_level(model):
    X_train_total = pd.read_csv('/Users/DYM/PycharmProjects/Movie-prediction/ProcessedData/base_result/StackingDataTrain_orig2.csv')
    y_train = pd.read_csv('Dev_Set/dev_set_groundtruth_and_trailers.csv')['goodforairplane']
    visual_train = pd.read_csv('/Users/DYM/PycharmProjects/Movie-prediction/ProcessedData/VisualData.csv')
    X_test_total = pd.read_csv('/Users/DYM/PycharmProjects/Movie-prediction/ProcessedData/test_set/base_result/StackingDataTest_orig2.csv')
    y_test = pd.read_csv('/Users/DYM/PycharmProjects/Movie-prediction/Test_Set/TestLabel.csv')['goodforairplanes']
    visual_test = pd.read_csv('/Users/DYM/PycharmProjects/Movie-prediction/ProcessedData/test_set/VisualData.csv')
    text = pd.read_csv('/Users/DYM/PycharmProjects/Movie-prediction/ProcessedData/TextualData.csv')

    # This is for complex metadata
    split = [range(3,8),range(8,12),range(12,19),range(23,28)]
    split1 = [range(23,28)]

    #this is for simple metadata
    # split = [range(0,3),range(3,6),range(6,10),range(10,17),range(17,26)]
    #split1 = [range(3,26)]
    for i in split1:
        print('=================================\n================================')
        X_train = X_train_total[X_train_total.columns[i]]
        X_test = X_test_total[X_test_total.columns[i]]

        param = gridcv(model)
        clf = model()
        metric = ['f1','accuracy','precision']
        print('==================',model,'=============')
        for i in metric:
            grid = GridSearchCV(clf,param_grid= param,cv=10,scoring = i)
            grid.fit(X_train, y_train)
            print("The best parameters are %s with a score of %0.3f"
                  % (grid.best_params_, grid.best_score_))

            pre = grid.predict(X_test)
            f1, racall, precision = f1_score(y_test, pre), recall_score(y_test, pre), precision_score(y_test, pre)
            print(i,': ',f1, racall, precision)
            print(pre)


if __name__ == '__main__':
    # detail = loaddetail()
    # meta_level_train(detail,method='predict_proba')
    # create_stack_test(detail,method='predict_proba')
    meta_level(SVC)
    print('=================================\n================================')
    meta_level(AdaBoostClassifier)
    print('=================================\n================================')
    meta_level(GradientBoostingClassifier)
    # print('=================================\n================================')
    # meta_level(XGBClassifier)

    # def meta_level_classifier():
    #     X_train = pd.read_csv(
    #         '/Users/DYM/PycharmProjects/Movie-prediction/ProcessedData/base_result/StackDataTrain_orig1.csv')
    #     y_train = pd.read_csv('Dev_Set/dev_set_groundtruth_and_trailers.csv')['goodforairplane']
    #     visual_train = pd.read_csv('/Users/DYM/PycharmProjects/Movie-prediction/ProcessedData/VisualData.csv')
    #     X_test = pd.read_csv(
    #         '/Users/DYM/PycharmProjects/Movie-prediction/ProcessedData/test_set/base_result/StackDataTest_orig1.csv')
    #     y_test = pd.read_csv('/Users/DYM/PycharmProjects/Movie-prediction/Test_Set/TestLabel.csv')['goodforairplanes']
    #     visual_test = pd.read_csv('/Users/DYM/PycharmProjects/Movie-prediction/ProcessedData/test_set/VisualData.csv')
    #     text = pd.read_csv('/Users/DYM/PycharmProjects/Movie-prediction/ProcessedData/TextualData.csv')
    #     visual_train = pd.DataFrame(MinMaxScaler().fit_transform(visual_train))
    #     visual_test = pd.DataFrame(MinMaxScaler().fit_transform(visual_test))
    #     X_combing_train = pd.concat([X_train, visual_train], axis=1)
    #     X_combing_test = pd.concat([X_train, visual_test], axis=1)
    #     clf = SVC()
    #     ada = AdaBoostClassifier()
    #     param_grid = {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.001, ]}
    #     # param_grid={"C":[0.01,0.1, 1, 10], "gamma": [1, 0.1, 0.01]}
    #     grid = GridSearchCV(ada, param_grid=param_grid, cv=10, scoring='f1')
    #     grid.fit(X_train, y_train)
    #     print("The best parameters are %s with a score of %0.3f"
    #           % (grid.best_params_, grid.best_score_))
    #
    #     pre = grid.predict(X_test)
    #     f1, racall, precision = f1_score(y_test, pre), recall_score(y_test, pre), precision_score(y_test, pre)
    #     print(f1, racall, precision)
    #     print(pre)

