import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from sklearn.decomposition import PCA
import os


def obtaintitle(path):
    filelist_csv = os.listdir(path)
    filelist = []
    for file in filelist_csv:
        f = os.path.splitext(file)[0]
        filelist.append(f)
    return filelist, filelist_csv

def obtainvector(n):
    filelist, filelist_csv = obtaintitle('Dev_Set/vis_descriptors')
    df=pd.read_csv('Dev_Set/vis_descriptors/' + str(filelist_csv[n]), header=None)
    data = np.array(df)
    data = data.reshape(1, 1652)
    df = pd.DataFrame(data)
    return df

def obtainmeta(n):
    filelist, filelist_xml = obtaintitle('Dev_Set/XML')
    tree = ET.ElementTree(file='Dev_Set/XML/'+str(filelist_xml[n]))
    root = tree.getroot()
    metadata = []
    metadata.append(root[0].attrib)
    df = pd.DataFrame(metadata)
    return df

def obtainv3(n):
    model = InceptionV3(weights='imagenet', include_top=None, pooling='avg')
    filelist, filelist_jpj = obtaintitle('Dev_Set/Posters')
    img_path = 'Dev_Set/Posters/'+filelist_jpj[n]
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = pd.DataFrame(
        model.predict(x)
    )
    return features
def obtainPca(n):
    filelist, filelist_csv = obtaintitle('Dev_Set/audio_descriptors')
    df = pd.read_csv('Dev_Set/audio_descriptors/'+str(filelist_csv[n]), header=None)
    df = df.fillna(0)
    df = df.T
    pca = PCA(n_components=1)
    newData = pca.fit_transform(df)
    newData = newData.T
    df = pd.DataFrame(newData)
    return df

def create_data(method,file_name):
    # title = pd.read_csv('Dev_Set/dev_set_groundtruth_and_trailers.csv')
    # title = title.drop(['movie'], axis= 1)
    # title = title.drop(['trailer'], axis= 1)

    df = method(0)

    for i in range(1,95):
        df1 = method(i)
        df = pd.concat([df,df1])
        print(i , str(method) + 'has been concated')
    df.index = range(0,95)
    # df2 = pd.concat([title,df],axis=1)
    df.to_csv('ProcessedData/'+file_name, index=False, header=True)
    return df

def create_textdata(file_name):
    df = pd.read_csv('Dev_Set/text_descriptors/tdf_idf_dev_orig.csv', header=None)
    df = df.T
    df.to_csv('ProcessedData/' + file_name, header=True,index = False)
    return df

def Process_Metadata():
    df_orig = pd.read_csv('processedData/MetaCombine.csv')
    df=df_orig.drop(['BoxOffice','DVD','Production','Website','actors','awards',
            'country','director','imdbID','plot','poster','released','released',
            'type','writer','year','tomatoConsensus','title','year'],axis=1)
    for i in range(len(df.runtime)):
        string = df.runtime[i]
        df.runtime[i] = string.replace(' min','')

    # deal with awards category
    df.runtime = df.runtime.astype(float)
    df_orig.awards = df_orig.awards.fillna('unknown')
    df_orig['IsOscar'] = 1
    df_orig['IsWin'] = 1
    for i in range(0, 318):
        df_orig.awards[i] = df_orig.awards[i].lower()
        if df_orig.awards[i].find('oscar') == -1:
            df_orig['IsOscar'][i]= 0
        if df_orig.awards[i].find('win') == -1:
            df_orig['IsWin'][i] = 0
    df['IsOscar'] = df_orig['IsOscar']
    df['IsWin'] = df_orig['IsWin']

    #deal with country category
    df_orig['IsForeign'] = 1
    df_orig['IsUSA'] = 1
    for i in range(0, 318):
        if df_orig.country[i].find('USA') != -1 or df_orig.country[i].find('UK') !=-1:
            df_orig['IsForeign'][i]=0
        else:
            df_orig['IsForeign'][i] == 1
        if df_orig.country[i].find('USA') == -1:
            df_orig['IsUSA'][i] =0
    df['IsForeign'] = df_orig['IsForeign']
    df['IsUSA'] = df_orig['IsUSA']

    #deal with genre category
    df_orig['IsAdventure']=1
    df_orig['IsAnimation']=1
    df_orig['IsComedy']=1
    df_orig['IsCrime']=1
    df_orig['IsDrama']=1
    df_orig['IsAction']=1
    df_orig['IsThriller']=1

    for i in range(0, 318):
        if df_orig.genre[i].find('IsAdventure') == -1:
            df_orig['IsAdventure'][i] =0
        if df_orig.genre[i].find('IsAnimation') == -1:
            df_orig['IsAnimation'][i] =0
        if df_orig.genre[i].find('IsComedy') == -1:
            df_orig['IsComedy'][i] =0
        if df_orig.genre[i].find('IsDrama') == -1:
            df_orig['IsDrama'][i] =0
        if df_orig.genre[i].find('IsAction') == -1:
            df_orig['IsAction'][i] =0
        if df_orig.genre[i].find('IsThriller') == -1:
            df_orig['IsThriller'][i] = 0

    df['IsAdventure']=df_orig['IsAdventure']
    df['IsAnimation']=df_orig['IsAnimation']
    df['IsComedy'] =df_orig['IsComedy']
    df['IsCrime'] =df_orig['IsCrime']
    df['IsDrama'] =df_orig['IsDrama']
    df['IsAction'] =df_orig['IsAction']
    df['IsThriller'] =df_orig['IsThriller']

    #deal with Language
    df_orig['IsENMix']=1
    df_orig['IsEN']=1
    df_orig['IsNonEN']=1
    for i in range(0, 318):
        if df_orig.language[i].find('English') == -1:
            df_orig['IsEN'][i]=0
        if df_orig.language[i].find('English') != -1 & len(df_orig.language[i])==7:
            df_orig['IsENMix'][i]=0
        if df_orig.language[i].find('English') != -1:
            df_orig['IsNonEN'][i]=0
    df['IsENMix']= df_orig['IsENMix']
    df['IsEN'] = df_orig['IsEN']
    df['IsNonEN'] = df_orig['IsNonEN']
    #deal with Tomato Rotten
    df_orig['NonTomato'] = (df['tomatoMeter'].isnull()) * 1
    df['NonTomato'] = (df['tomatoMeter'].isnull()) * 1
    df_orig['NonTomatoUser']=(df['tomatoUserMeter'].isnull()) * 1
    df['NonTomatoUser'] = (df['tomatoUserMeter'].isnull()) * 1
    #deal with year
    df_orig['age'] = 2017 - df_orig['year']
    df['age'] = 2017 - df_orig['year']

    #deal with missing value
    feature_names = df.columns[0:].tolist()
    print('these feature contain missing value ')
    for feature in feature_names:
        if len(df[feature][df[feature].isnull()]) > 0:
            print(feature)

    for feature in ['tomatoImage','rated']:
        df[feature][df[feature].isnull()] = 'U0'

    for feature in feature_names:
        if len(df[feature][df[feature].isnull()]) > 0:
            df[feature][df[feature].isnull()]=df[feature].median()
    df1=pd.get_dummies(df)
    df1.to_csv('ProcessedData/ProcessedMetaCombining.csv',index=False)
    return

def process_text():
    name_train = pd.read_csv('/Users/DYM/PycharmProjects/Movie-prediction/original_textual/tdf_idf_dev.csv')
    name_train = name_train.columns.tolist()
    name_test = pd.read_csv('/Users/DYM/PycharmProjects/Movie-prediction/original_textual/tdf_idf_test.csv')
    name_test = name_test.columns.tolist()
    differ = [elem for elem in name_train if elem not in name_test]
    dict = {k:[0]*128 for k in range(0,len(name_train))}
    comp = pd.DataFrame(dict)
    comp.index = range(95, 223)
    comp.columns = comp.columns.astype("str")
    text_train = pd.read_csv('/Users/DYM/PycharmProjects/Movie-prediction/ProcessedData/TextualData.csv')
    text_test = pd.read_csv('/Users/DYM/PycharmProjects/Movie-prediction/ProcessedData/test_set/TextualData.csv')
    text_test.columns = name_test
    text_train = pd.concat([text_train,comp],axis=0)
    text_train.columns = name_train
    test_new = pd.DataFrame({k: [0] * 223 for k in differ})
    text_test_new = pd.concat([text_test, test_new], axis=1)
    name_test_new = text_test_new.columns.tolist()
    text_train_new = pd.DataFrame()
    for i in name_test_new:
        if i in name_train:
            text_train_new[i] = text_train[i]
        else:
            text_train_new[i] = [0]*95
    text_train_new.columns = range(0,8039)
    text_test_new.columns = range(0,8039)
    text_train_new.to_csv('/Users/DYM/PycharmProjects/Movie-prediction/ProcessedData/TextData.csv',index=False)
    text_test_new.to_csv('/Users/DYM/PycharmProjects/Movie-prediction/ProcessedData/test_set/Textdata.csv',index=False)

def divide_meta():
    df = pd.read_csv('ProcessedData/ProcessedMetaCombining.csv')
    df1 = df.iloc[0:95, :]
    df2 = df.iloc[95:, :]
    df1.to_csv('ProcessedData/ProcessedMeta.csv',index=False)
    df2.to_csv('ProcessedData/test_set/ProcessedMeta.csv', index=False)

if __name__ == '__main__':

     #create_data(obtainv3,'V3Data.csv')

    # create_data(obtainmeta,file_name='MetaData.csv')

    #create_data(obtainvector,file_name='VisualData.csv')

    # create_textdata('TextualData.csv')

    # create_data(obtainPca, file_name='AudioData.csv')

    Process_Metadata()
    divide_meta()
    #  process_text()