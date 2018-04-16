#%%
# 库加载
import pandas as pd
import numpy as np 
import sys
import os
from pandas import Series
from functools import reduce
from pyspark import SparkContext 
from pyspark.sql import SQLContext 
from collections import Counter
from __future__ import division

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
    ic = intersection_count(l1,l2)
    uc = union_count(l1,l2)
    return(ic, uc, ic/uc)

#%%
# 数据预处理
data_raw = pd.read_csv("../Data/Input/company_tag_data_raw", sep='\t', dtype={"comp_id":str})
cols = ["comp_id", "comp_full_name", "label_name", "classify_id", "label_type", "label_type_num", "src_tags"]
data_raw = data_raw[cols]
concept_tags = data_raw[data_raw.classify_id!=4].reset_index(drop=True)
#%%
level_data_raw = pd.read_csv("../Data/Input/label_code_relation", sep='\t', dtype={"label_root_id":str, "label_note_id":str})
tag_code_dict = pd.concat([level_data_raw.label_note_name, level_data_raw.label_root_name]).drop_duplicates().reset_index(drop=True)
tag_code_dict.name = "label_name"
tag_code_dict = tag_code_dict.reset_index()
tag_code_dict.rename(index=str, columns={"index": "tag_code"}, inplace=True)
length = len(str(len(tag_code_dict)))
tag_code_dict.tag_code = tag_code_dict.tag_code.apply(lambda x: str(x).zfill(length))
concept_tags_with_code = pd.merge(concept_tags, tag_code_dict, how='left', left_on='label_name', right_on='label_name') \
    .dropna(how='any')
tag_code_dict

#%%
# （前期尝试，目前倾向于下面的一次性计算方案）
# 单独计算两两标签共同出现在同一公司标签列表中的频次
sc = SparkContext.getOrCreate()
sqlContext=SQLContext(sc)
tags_by_comp = concept_tags_with_code[["comp_id","tag_code"]].groupby("comp_id").agg(pinjie).reset_index()
tags_by_comp = tags_by_comp[tags_by_comp.tag_code.apply(lambda x: len(x.split(","))>=2)]
tags_by_comp["tag_couple"] = tags_by_comp.tag_code.apply(lambda x: tag_couple(list(set(x.split(','))), length))
spark_tags_by_comp = sqlContext.createDataFrame(tags_by_comp)
spark_result_rdd = spark_tags_by_comp.rdd. \
    flatMap(lambda x: map(lambda x: (x,1),x[2].split(","))).reduceByKey(lambda x,y : x + y)
spark_result_df = sqlContext.createDataFrame(spark_result_rdd, schema=["tag_link","count"])
tag_link_df = sqlContext.createDataFrame(spark_result_df.rdd. \
    map(lambda x: (x[0].split("-")[0], x[0].split("-")[1], x[1])), ["tag1","tag2","count"])
tag_link_py_df = tag_link_df.toPandas()
# tag_link_df.collect()

#%%
# 一次性统计两两标签覆盖公司列表的交集、并集数目及比例
comps_by_tag = concept_tags_with_code[["tag_code","comp_id"]].groupby("tag_code").agg(pinjie).reset_index()
comps_by_tag.tag_code = comps_by_tag.tag_code.apply(lambda x: x.zfill(length))
comps_by_tag_df =  sqlContext.createDataFrame(comps_by_tag)
comps_by_tag_df2 = comps_by_tag_df.withColumnRenamed("tag_code","tag_code2").withColumnRenamed("comp_id","comp_id2")
all_relation = comps_by_tag_df.crossJoin(comps_by_tag_df2).filter("tag_code != tag_code2")
statistic_result_rdd = all_relation.rdd \
    .map(lambda x: ((x[0],x[2]) if int(x[0])>=int(x[2]) else (x[2],x[0])) + final_count(x[1].split(","),x[3].split(","))).distinct()
statistic_result_df = sqlContext.createDataFrame(statistic_result_rdd, schema=["tag1","tag2","intersection","union","percentage"]) \
    .filter("intersection != 0")
statistic_result_py_df = statistic_result_df.toPandas()
statistic_result_py_df.to_csv("../Data/Output/tag_relation_value.csv", index=False)
print("Data saved!")
# statistic_result_df.show()

#%%
# 概念标签之间的层级关系
label_chains = level_data_raw[level_data_raw.label_type_root-level_data_raw.label_type_note==-1].reset_index(drop=True) \
    .rename(index=str, columns={"label_note_name": "label_node_name"}, inplace=False)[["label_node_name", "label_root_name"]]
tag_code_root = tag_code_dict.rename(index=str, columns={"tag_code": "root_code", "label_name": "root_name"}, inplace=False)
tag_code_node = tag_code_dict.rename(index=str, columns={"tag_code": "node_code", "label_name": "node_name"}, inplace=False)
label_chains = label_chains.merge(tag_code_node, how='left', left_on='label_node_name', right_on='node_name') \
    .merge(tag_code_root, how='left', left_on='label_root_name', right_on='root_name')#[["node_code", "root_code"]]
label_chains
#print(label_chains)

#%%
# 公司和标签关系


#%%
sc.stop()
#%%