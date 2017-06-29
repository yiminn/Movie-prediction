import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from sklearn.decomposition import PCA
import os
def obtaintitle(path):
    filelist_ext = os.listdir(path)
    filelist = []
    for file in filelist_ext:
        f = os.path.splitext(file)[0]
        filelist.append(f)
    return filelist, filelist_ext

def obtainvector(n):
    filelist, filelist_csv = obtaintitle('Test_Set/vis_descriptors')
    df=pd.read_csv('Test_Set/vis_descriptors/' + str(filelist_csv[n]), header=None)
    data = np.array(df)
    data = data.reshape(1, 1652)
    df = pd.DataFrame(data)
    return df

def obtainmeta(n):
    filelist, filelist_xml = obtaintitle('Test_Set/XML')
    tree = ET.ElementTree(file='Test_Set/XML/'+str(filelist_xml[n]))
    root = tree.getroot()
    metadata = []
    metadata.append(root[0].attrib)
    df = pd.DataFrame(metadata)
    return df

def obtainv3(n):
    model = InceptionV3(weights='imagenet', include_top=None, pooling='avg')
    filelist, filelist_jpj = obtaintitle('Test_Set/Posters')
    img_path = 'Test_Set/Posters/'+filelist_jpj[n]
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = pd.DataFrame(
        model.predict(x)
    )
    return features

def obtainPca(n):
    filelist, filelist_csv = obtaintitle('Test_Set/audio_descriptors')
    df = pd.read_csv('/Users/DYM/PycharmProjects/MasterFYP/Test_Set/audio_descriptors/'+str(filelist_csv[n]), header=None)
    df = df.fillna(0)
    df = df.T
    pca = PCA(n_components=1)
    newData = pca.fit_transform(df)
    newData = newData.T
    df = pd.DataFrame(newData)
    return df

def create_data(method,file_name):
    title = pd.read_csv('Test_Set/test_set_trailers.csv')
    title = title.drop(['movie'], axis= 1)
    title = title.drop(['trailer'], axis= 1)

    df = method(0)

    for i in range(1,223):
        df1 = method(i)
        df = pd.concat([df,df1])
        print(i,'has been concacted')
    df.index = range(0,223)
    df2 = pd.concat([title,df],axis=1)
    df2.to_csv('ProcessedData/'+file_name, index=False, header=True)
    return df2

def create_textdata(file_name):
    title = pd.read_csv('Test_Set/test_set_trailers.csv')
    title = title.drop(['movie'], axis= 1)
    title = title.drop(['trailer'], axis= 1)
    df = pd.read_csv('/Users/DYM/PycharmProjects/MasterFYP/Test_Set/text_descriptors/tdf_idf_test.csv', header=None)
    df = df.T
    df2 = pd.concat([title, df], axis=1)
    df2.to_csv('ProcessedData/'+file_name, index=False, header=True)
    return df2

if __name__ == '__main__':
    # create_data(obtainvector,'VisualDataTest.csv')

    # create_data(obtainmeta,'MetaDataTest.csv')

    # create_data(obtainv3,'V3DataTest.csv')

      #create_textdata('TextualDataTest.csv')

    create_data(obtainPca, 'AudioDataTest.csv')

#print(data)