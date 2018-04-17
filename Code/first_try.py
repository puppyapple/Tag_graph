#%%
import pandas as pd
import numpy as np 
import sys
import os
from pandas import Series
from functools import reduce
from pyspark import SparkContext 
from pyspark.sql import SQLContext 
from collections import Counter

#%%
def pinjie(arr):
    return ",".join(arr)

#%%
def tag_couple(l, length):
    result = []
    for i in range(len(l)):
        for j in range(i+1, len(l)):
            tmp = l[i].zfill(length) + "-" + l[j].zfill(length) if int(l[i]) >= int(l[j]) \
                else l[j].zfill(length) + "-" + l[i].zfill(length)
            result.append(tmp)
    return ",".join(result)

#%%
data_raw = pd.read_csv("Desktop/Innotree/Graph/Data/company_tag_data_raw", sep='\t', dtype={"comp_id":str})
cols = ["comp_id", "comp_full_name", "label_name", "classify_id", "label_type", "label_type_num", "src_tags"]
data_raw = data_raw[cols]
concept_tags = data_raw[data_raw.classify_id!=4].reset_index(drop=True)
# concept_tags
#%%
tag_code_dict = concept_tags.label_name.drop_duplicates().reset_index(drop=True)
tag_code_dict = tag_code_dict.reset_index()
tag_code_dict.rename(index=str, columns={"index": "tag_code"}, inplace=True)
tag_code_dict.tag_code = tag_code_dict.tag_code.apply(lambda x: str(x))
# print(tag_code_dict.dtypes)
#%%
length = len(str(len(tag_code_dict)))
concept_tags_with_code = pd.merge(concept_tags, tag_code_dict, how='left', left_on='label_name', right_on='label_name')
# print(concept_tags_with_code)
tags_by_comp = concept_tags_with_code[["comp_id","tag_code"]].groupby("comp_id").agg(pinjie).reset_index()
tags_by_comp["tag_couple"] = tags_by_comp.tag_code.apply(lambda x: tag_couple(x.split(','), length))
# tags_by_comp
#%%
result_list = reduce(lambda x, y: x + "," +  y, tags_by_comp.tag_couple)
# len(set(result_list))
#%%
# len(result_list)
tag_couple_count = Counter(result_list.split(","))
tag_couple_count

#%%
len(tag_couple_count)

#%%
sc = SparkContext()
sqlContext=SQLContext(sc)


#%%
spark_tags_by_comp = sqlContext.createDataFrame(tags_by_comp)
spark_result = spark_tags_by_comp.rdd.flatMap(lambda x: map(lambda x: (x,1),x[2].split(","))).reduceByKey(lambda x,y : x + y)
spark_result.collect()

# test.reduceByKey(lambda x,y : x + "," + y).collect()