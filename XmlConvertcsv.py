import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import os

def obtaintitle(path):
    filelist_xml = os.listdir(path)
    filelist = []
    for file in filelist_xml:
        f = os.path.splitext(file)[0]
        filelist.append(f)
    return filelist, filelist_xml

def obtainmeta(n):
    filelist, filelist_xml = obtaintitle('Dev_Set/XML')
    tree = ET.ElementTree(file='Dev_Set/XML/'+str(filelist_xml[n]))
    root = tree.getroot()
    metadata = []
    metadata.append(root[0].attrib)
    df = pd.DataFrame(metadata)
    return df


df = obtainmeta(0)

title = pd.read_csv('Dev_Set/dev_set_groundtruth_and_trailers.csv')
title = title.drop(['movie'], axis= 1)
title = title.drop(['trailer'], axis= 1)

for i in range(1,95):
    df1 = obtainmeta(i)
    df = pd.concat([df,df1])
df.index = range(0,95)
df2 = pd.concat([title,df],axis=1)
df2.to_csv('ProcessedData/MetaData.csv', index=True, header=True)

