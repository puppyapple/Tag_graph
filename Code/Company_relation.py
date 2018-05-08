#%%
# 库加载
from __future__ import division
import pandas as pd
import numpy as np 
import sys
import os

from pandas import Series
from functools import reduce
from pyspark import SparkContext 
from pyspark.sql import SQLContext 
from collections import Counter
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
print(os.getcwd())
# 自定义函数
def pinjie(arr):
    return ",".join(arr)


def tag_couple(l, length):
    result = []
    for i in range(len(l)):
        for j in range(i+1, len(l)):
            tmp = l[i].zfill(length) + "-" + l[j].zfill(length) if int(l[i]) >= int(l[j]) \
                else l[j].zfill(length) + "-" + l[i].zfill(length)
            result.append(tmp)
    return ",".join(result)

def union_count(l1, l2):
    return len(set(set(l1).union(set(l2))))

def intersection_count(l1, l2):
    return len(set(set(l1).intersection(set(l2))))

def final_count(l1, l2):
    ic = intersection_count(l1, l2)
    uc = union_count(l1, l2)
    return(ic, uc, ic/uc)

#%%
# 数据预处理
sc = SparkContext.getOrCreate()
sqlContext=SQLContext(sc)
data_raw = pd.read_csv("../Data/Input/company_tag_data_raw", sep='\t', dtype={"comp_id":str})
cols = ["comp_id", "comp_full_name", "label_name", "classify_id", "label_type", "label_type_num", "src_tags"]
data_raw = data_raw[cols]
concept_tags = data_raw[data_raw.classify_id != 4].reset_index(drop=True)
concept_tags.label_name = concept_tags[["label_name", "label_type_num", "src_tags", "label_type"]] \
    .apply(lambda x: x[2].split("#")[x[1]-1].split("-")[max(x[3]-2, 0)] + ":" + x[0], axis=1)
#%%
# 概念关系表字典
level_data_raw = pd.read_csv("../Data/Input/label_code_relation", sep='\t', dtype={"label_root_id":str, "label_note_id":str})
tag_code_dict = pd.concat([level_data_raw.label_note_name, level_data_raw.label_root_name]).drop_duplicates().reset_index(drop=True)
tag_code_dict.name = "label_name"
tag_code_dict = tag_code_dict.reset_index()
tag_code_dict.rename(index=str, columns={"index": "tag_code"}, inplace=True)
length = len(str(len(tag_code_dict)))
tag_code_dict.tag_code = tag_code_dict.tag_code.apply(lambda x: str(x).zfill(length))