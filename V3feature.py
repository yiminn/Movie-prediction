from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
import numpy as np
import pandas as pd
import LoadData



def obtainv3(n):
    model = InceptionV3(weights='imagenet',include_top=None,pooling='avg')
    filelist, filelist_jpj = LoadData.obtaintitle('Dev_Set/Posters')

    img_path = 'Dev_Set/Posters/'
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = pd.DataFrame(
        model.predict(x)
        )

    return features


title = pd.read_csv('Dev_Set/dev_set_groundtruth_and_trailers.csv')
title = title.drop(['movie'], axis=1)
title = title.drop(['trailer'], axis=1)

df = obtainv3(0)

for i in range(1, 95):
    df1 = obtainv3(i)
    df = pd.concat([df, df1])
df.index = range(0, 95)
df2 = pd.concat([title, df], axis=1)
df2.to_csv('ProcessedData/' + file_name, index=True, header=True)