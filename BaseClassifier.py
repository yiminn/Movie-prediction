import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score,cross_val_predict,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import  MultinomialNB
from xgboost.sklearn import XGBClassifier
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


def modelling(data,model):
    X_train, y_train = obtaindata(data)
    if model == SVC:
        clf = SVC(probability=True)
    else:
        clf =model()
    scores1 = cross_val_score(clf, X_train, y_train, cv=10,scoring='f1')
    scores2 = cross_val_score(clf, X_train, y_train, cv=10,scoring='precision')
    scores3 = cross_val_score(clf, X_train, y_train, cv=10,scoring='recall')
    f1_score = scores1.mean()
    precision_score = scores2.mean()
    recall_score = scores3.mean()
    _,name = creat_name(data, model)
    print(name)
    print('f1: %.3f \nprecesion: %f \nrecall: %f' %(f1_score, precision_score, recall_score))
    print('\n-------------------------------\n')
    return  f1_score,precision_score,recall_score,



if __name__ == '__main__':
    base_model = [XGBClassifier,DecisionTreeClassifier,KNeighborsClassifier, MultinomialNB, LogisticRegression,
                  SVC, BaggingClassifier, RandomForestClassifier, AdaBoostClassifier,
                  GradientBoostingClassifier]
    typeDataset = ('AudioData', 'TextData', 'V3Data', 'VisualData','ProcessedMeta')

    metrics = {}
    for data in typeDataset:
        detail_set = []
        print('====================',data,'=====================================')
        for model in base_model:
            try:
                detail = []
                combing, model_name = creat_name(data, model)
                f1,precision,recall = modelling(data,model)
                if f1>0.5 and precision>0.5 and recall>0.5:
                    if model  == SVC:
                        detail = [model_name, {'probability': True}, f1,precision,recall]
                    else:
                        detail = [model_name, {}, f1, precision, recall]
                    detail_set.append(detail)
            except ValueError:
                print(combing, ': has error\n-------------------------------\n')
                pass
            metrics[data] = detail_set
    print(metrics)
    file1=open('/Users/DYM/PycharmProjects/Movie-prediction/Data_Record/type_model1.txt','wb')
    pickle.dump(metrics,file1)
    file1.close()








    #####################################LVW
    # wrapper = LVW()
    # print('V3Data')
    # for clf in base_model:
    #     clf=clf()
    #     X_train,  y_train = obtaindata('V3Data')
    #     best_features, optimized_accuracy = wrapper.lvw(X=X_train, y=y_train, clf=clf, iteration=1000,opti_obej='f1')
    #     print(clf,optimized_accuracy)