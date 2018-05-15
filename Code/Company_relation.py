#%%
# 库加载
from __future__ import division
import pandas as pd
import numpy as np 
import sys
import os
import operator
from pandas import Series
from functools import reduce
from pyspark import SparkContext 
from pyspark.sql import SQLContext 
from collections import Counter
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from numpy import linalg

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

def count_product(l1, l2):
    return np.sqrt(len(set(l1)) * len(set(l2)))

def final_count(l1, l2):
    ic = intersection_count(l1, l2)
    uc = union_count(l1, l2)
    return(ic, uc, ic/uc)

def final_count2(l1, l2):
    ic = intersection_count(l1, l2)
    cp = count_product(l1, l2)
    return(ic, cp, ic/cp)

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

#%%
# 按标签聚合公司
comps_by_tag = concept_tags[["label_name", "comp_id"]].groupby("label_name").agg(pinjie).reset_index()
comps_by_tag["label_name_father"] = comps_by_tag.label_name.apply(lambda x: x.split(":")[0])
comps_by_tag["label_name"] = comps_by_tag.label_name.apply(lambda x: x.split(":")[1])
comps_by_tag_rough = comps_by_tag.groupby("label_name").agg(pinjie).reset_index()[["label_name", "comp_id"]]
concept_tags


#%%
# 为标签分组表赋上标签代码
comps_by_tag_rough_with_code = comps_by_tag_rough.merge(tag_code_dict, how='left', left_on='label_name', right_on='label_name') \
    .dropna(how='any')
tag_code_dict_2 = tag_code_dict.copy()
tag_code_dict_2.columns = ["tag_code_father", "label_name_father"]
comps_by_tag_with_code = comps_by_tag.merge(tag_code_dict, how='left', left_on='label_name', right_on='label_name') \
    .merge(tag_code_dict_2, how='left', left_on='label_name_father', right_on='label_name_father') \
    .dropna(how='any')
comps_by_tag_with_code


#%%
# 概念标签之间的层级关系（tag1 belongs to tag2）
label_chains_raw = level_data_raw.reset_index(drop=True) \
    .rename(index=str, columns={"label_note_name":"label_node_name", "label_type_note":"label_type_node"}, inplace=False)
tag_code_root = tag_code_dict.rename(index=str, columns={"tag_code":"root_code", "label_name":"root_name"}, inplace=False)
tag_code_node = tag_code_dict.rename(index=str, columns={"tag_code":"node_code", "label_name":"node_name"}, inplace=False)
label_chains_full = label_chains_raw.merge(tag_code_node, how='left', left_on='label_node_name', right_on='node_name') \
    .merge(tag_code_root, how='left', left_on='label_root_name', right_on='root_name')[["node_code", "root_code", "label_type_node", "label_type_root"]]
label_chains = label_chains_full[level_data_raw.label_type_root-level_data_raw.label_type_note==-1]
label_chains
# 考虑将子标签公司数目占父标签公司数目的比例作为边强度
node_tag_companies = comps_by_tag_with_code[["tag_code", "tag_code_father", "comp_id"]] \
    .rename(index=str, columns={"tag_code":"node_code", "tag_code_father": "root_code", "comp_id":"node_comps"}, inplace=False)
root_tag_companies = comps_by_tag_rough_with_code[["tag_code", "comp_id"]] \
    .rename(index=str, columns={"tag_code":"root_code", "comp_id":"root_comps"}, inplace=False)
label_chains = label_chains.merge(node_tag_companies, how='left', on=["node_code", "root_code"]) \
    .merge(root_tag_companies, how='left', left_on='root_code', right_on='root_code')
label_chains["proportion"] = label_chains.node_comps.apply(lambda x: len(set(x.split(","))) if isinstance(x, str) else 0) \
    /label_chains.root_comps.apply(lambda x: len(set(x.split(","))) if isinstance(x, str) else 0)
#%%
proportion_reset = label_chains.groupby(["root_code", "node_code"]).agg({"proportion": "sum"}) \
    .groupby(level=0).apply(lambda x: x/float(x.sum())).reset_index()
label_chains = label_chains.drop(['proportion'], axis=1).merge(proportion_reset, how='left', on=['root_code', 'node_code'])
label_chains.drop_duplicates(inplace=True)
label_chains.fillna(0.0, inplace=True)
label_chains_new = label_chains[["node_code", "root_code", "proportion"]].copy()
label_chains_new

#%%
# 一次性统计两两标签覆盖公司列表的交集、并集数目及比例
comps_by_tag_df =  sqlContext.createDataFrame(comps_by_tag_rough_with_code[["tag_code", "comp_id"]])
comps_by_tag_df2 = comps_by_tag_df.withColumnRenamed("tag_code", "tag_code2").withColumnRenamed("comp_id", "comp_id2")
all_relation = comps_by_tag_df.crossJoin(comps_by_tag_df2).filter("tag_code != tag_code2")
statistic_result_rdd = all_relation.rdd \
    .map(lambda x: ((x[0],x[2]) if int(x[0])>=int(x[2]) else (x[2],x[0])) + final_count(x[1].split(","),x[3].split(","))).distinct()
# statistic_result_rdd.collect()
statistic_result_df = sqlContext.createDataFrame(statistic_result_rdd, schema=["tag1", "tag2", "intersection", "union", "percentage"]) \
    .filter("intersection != 0")
statistic_result_py_df = statistic_result_df.toPandas()
statistic_result_py_df["tag_link"] = statistic_result_py_df.tag1 + "-" + statistic_result_py_df.tag2
# statistic_result_py_df
# 去掉标签关系结果中本已属于同一链条的数据
# 将数值归一化到(0,1]
label_chains_link = label_chains_full[["node_code", "root_code"]] \
    .apply(lambda x: x[0] + "-" + x[1] if int(x[0])>=int(x[1]) else x[1]+ "-" + x[0], axis=1).reset_index()
label_chains_link.columns = ["mark", "node_root_link"]
label_chains_link.mark = label_chains_link.mark.apply(lambda x: 1)
print(len(label_chains_link))
#%%
tag_relation_with_link = statistic_result_py_df.merge(label_chains_link, how='left', left_on='tag_link', right_on='node_root_link')
tag_relation_value = tag_relation_with_link[tag_relation_with_link.mark != 1.0][["tag1", "tag2", "intersection", "union", "percentage"]]
tag_relation_value.percentage = tag_relation_value.percentage.apply(lambda x: np.log2(min(0.000001 + x, 1)))
target = tag_relation_value.percentage.values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0.001, 1))
scaler.fit(target)
tag_relation_value.percentage = scaler.transform(target)
tag_relation_value

#%%
# 标签关系值合并
same_link_relation = label_chains_new.rename(index=str, columns={"node_code": "tag1", "root_code": "tag2", "proportion": "link_value"})
non_link_relation = tag_relation_value[["tag1", "tag2", "percentage"]].rename(index=str, columns={"percentage": "link_value"})
link_relation_all = pd.concat([same_link_relation, non_link_relation])
link_relation_all

#%%
tag_code_list = list(tag_code_dict.tag_code)
link_relation_all2 = link_relation_all.copy()
link_relation_all2.columns = ["tag2", "tag1", "link_value"]
link_relation_all_with_reverse = pd.concat([link_relation_all, link_relation_all2])
link_value_by_tag = link_relation_all_with_reverse.groupby("tag1").apply(lambda x: dict(zip(x.tag2, x.link_value))).reset_index()
link_value_by_tag.columns = ["tag", "value_dict"]
link_value_by_tag.value_dict = link_value_by_tag.value_dict.apply(lambda x: np.array(list(map(lambda y: x.get(y, 0) ,tag_code_list))))
matrix_link_value = np.array(list(link_value_by_tag.value_dict))
matrix_link_value = matrix_link_value + np.eye(len(matrix_link_value), dtype=float)
matrix_link_value
#%%
# 按公司聚合标签
tags_by_comp = concept_tags.transform({"label_name": lambda x: x.split(":")[1], "comp_id": lambda x: x}) \
    .merge(tag_code_dict, how='left', left_on='label_name', right_on='label_name')[["tag_code", "comp_id"]] \
    .dropna(how='any') \
    .groupby("comp_id").agg(lambda x: dict(Counter(set(x)))).reset_index()
tags_by_comp["tag_code_vector"] = tags_by_comp.tag_code.apply(lambda x: np.array(list(map(lambda y: x.get(y, 0) ,tag_code_list))))
comp_vector_dict = dict(zip(tags_by_comp.comp_id, tags_by_comp.tag_code_vector.apply(lambda x: np.array(x))))
comp_vector_dict

#%%
company_info = data_raw[data_raw.classify_id != 4][["comp_id", "comp_full_name", "src_tags"]].drop_duplicates() \
    .groupby("comp_id").agg({"comp_full_name": lambda x: x.max(), "src_tags": lambda x: "#".join(x)}).reset_index()
comps_table = tags_by_comp[["comp_id", "tag_code_vector"]].merge(company_info, how="left", left_on="comp_id", right_on="comp_id")
comps_table

#%%
def coef1(v1, v2):
    return 1/(1 + (v1.sum() - v2.sum())**2)

def coef2(v1, v2):
    return v1.dot(v2.T)

def coef(v1, v2):
    return v1.dot(v2.T)/np.sqrt(1 + (v1.sum() - v2.sum())**2)

def top_N(company, n, matrix=matrix_link_value):
    comp_vector = comp_vector_dict[company]
    result = {k: comp_vector.dot(matrix).dot(v.T)*coef for k, v in comp_vector_dict.items()}
    sorted_result = sorted(result.items(), key=operator.itemgetter(1),reverse=True)
    return sorted_result[:n]

def top_N2(company, n, matrix=matrix_link_value):
    tmp = comps_table.copy()
    comp_vector = comp_vector_dict[company]
    tmp["score"] = tmp.tag_code_vector.apply(lambda x: np.array(x)). \
        apply(lambda x: comp_vector.dot(matrix).dot(x.T)*coef(comp_vector, x)) 
    return tmp.sort_values(by=["score"], ascending=False)[:n]


#%%
a = top_N2('10086147958042903546', 50)
a.to_excel("../Data/Output/a.xlsx")
a
