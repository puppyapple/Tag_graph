#%%
# 库加载
from __future__ import division
import pandas as pd
import numpy as np 
import sys
import os
import networkx as nx
import matplotlib.pyplot as plt

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
# 公司标签数据
sc = SparkContext.getOrCreate()
sqlContext=SQLContext(sc)
data_raw = pd.read_csv("../Data/Input/Tag_graph/company_tag_data_raw", sep='\t', dtype={"comp_id":str})
cols = ["comp_id", "comp_full_name", "label_name", "classify_id", "label_type", "label_type_num", "src_tags"]
data_raw = data_raw[cols]
# data_raw.dtypes
concept_tags = data_raw[data_raw.classify_id != 4].reset_index(drop=True)
concept_tags.label_name = concept_tags[["label_name", "label_type_num", "src_tags", "label_type"]] \
    .apply(lambda x: x[2].split("#")[x[1]-1].split("-")[max(x[3]-2, 0)] + ":" + x[0], axis=1)

#%%
# 概念关系表字典
level_data_raw = pd.read_csv("../Data/Input/Tag_graph/label_code_relation", sep='\t', dtype={"label_root_id":str, "label_note_id":str})
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
comps_by_tag_rough

#%%
# 为标签分组表赋上标签代码
concept_tags_with_code = concept_tags.merge(tag_code_dict, how='left', left_on='label_name', right_on='label_name') \
    .dropna(how='any')
comps_by_tag_rough_with_code = comps_by_tag_rough.merge(tag_code_dict, how='left', left_on='label_name', right_on='label_name') \
    .dropna(how='any')
comps_by_tag_rough_with_code
#%% 
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
    .merge(tag_code_root, how='left', left_on='label_root_name', right_on='root_name') \
    [["node_code", "label_node_name", "root_code", "label_root_name", "label_type_node", "label_type_root"]].drop_duplicates()
label_chains = label_chains_full[level_data_raw.label_type_root-level_data_raw.label_type_note==-1]
label_chains_full
#%% 
# 考虑将子标签公司数目占父标签公司数目的比例作为边强度
node_tag_companies = comps_by_tag_with_code[["tag_code", "tag_code_father", "comp_id"]] \
    .rename(index=str, columns={"tag_code":"node_code", "tag_code_father": "root_code", "comp_id":"node_comps"}, inplace=False)
root_tag_companies = comps_by_tag_rough_with_code[["tag_code", "comp_id"]] \
    .rename(index=str, columns={"tag_code":"root_code", "comp_id":"root_comps"}, inplace=False)
label_chains = label_chains.merge(node_tag_companies, how='left', on=["node_code", "root_code"]) \
    .merge(root_tag_companies, how='left', left_on='root_code', right_on='root_code')
label_chains["proportion"] = label_chains.node_comps.apply(lambda x: len(set(x.split(","))) if isinstance(x, str) else 0) \
    /label_chains.root_comps.apply(lambda x: len(set(x.split(","))) if isinstance(x, str) else 0)
proportion_reset = label_chains.groupby(["root_code", "node_code"]).agg({"proportion": "sum"}) \
    .groupby(level=0).apply(lambda x: x/float(x.sum())).reset_index()
label_chains = label_chains.drop(['proportion'], axis=1).merge(proportion_reset, how='left', on=['root_code', 'node_code'])
label_chains.drop_duplicates(inplace=True)
label_chains.fillna(0.0, inplace=True)

#%%
label_chains_single = label_chains[["node_code", "root_code", "proportion"]].copy()
# 这里认为同一链条上的相邻标签距离整体要近于非同一链条上的，因此追加一个距离“弱化”系数
weak_coef = 1/(6 - 1)
label_chains_single["proportion"] = label_chains_single["proportion"].apply(lambda x: (1 - x))
label_chains_reverse = label_chains_single.copy()
label_chains_single.columns = ["tag1", "tag2", "distance"]
label_chains_reverse.columns = ["tag2", "tag1", "distance"]
label_chains_final = pd.concat([label_chains_single, label_chains_reverse]).drop_duplicates()
# label_chains_new.columns = header_dict["level_tag_value"]
label_chains_final = label_chains_final[["tag1", "tag2", "distance"]]
label_chains_final.describe()

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

#%%
# 各个概念“树”的内部关联值，基于路径
code_table = pd.read_excel('../Data/Input/Tag_graph/编码表_4_26.xls', sheet_name=0).drop('标识', axis=1).fillna('')
code_table['list'] = code_table.apply(lambda x: ",".join([w for w in x if w != '']), axis=1)
l = list(range(0, 12))
l.append('list')
code_table.columns = l
label_list_by_tree = code_table[[0, 'list']].groupby(0).agg(lambda x: list(set(','.join(x).split(',')))).reset_index()
label_list_by_tree.columns = ['tree_name', 'label_list']
label_list_by_tree['label_count'] = label_list_by_tree.label_list.apply(lambda x: len(x))
tree_dict_raw = label_list_by_tree[label_list_by_tree.label_count > 1]
tree_dict = dict(zip(tree_dict_raw.tree_name, tree_dict_raw.label_list))
label_chains_full['chain_type_node'] = label_chains_full.label_node_name.apply(lambda x: [k for k, v in tree_dict.items() if x in v])
label_chains_full['chain_type_root'] = label_chains_full.label_root_name.apply(lambda x: [k for k, v in tree_dict.items() if x in v])
label_chains_full['chain_type'] = label_chains_full[['chain_type_node', 'chain_type_root']].apply(lambda x: set(x[0]).intersection(set(x[1])), axis=1)
label_chains_full['chain_type_count'] = label_chains_full.chain_type.apply(lambda x: len(x))
direct_link = label_chains_full[label_chains_full.label_type_root - label_chains_full.label_type_node == -1]
duplicated_chain_type = direct_link[direct_link.chain_type_count == 2]
seperated_1 = duplicated_chain_type.copy()
seperated_2 = duplicated_chain_type.copy()
seperated_1.chain_type = seperated_1.chain_type.apply(lambda x: list(x)[0])
seperated_2.chain_type = seperated_2.chain_type.apply(lambda x: list(x)[1])
seperated_3 = direct_link[direct_link.chain_type_count == 1].copy()
seperated_3.chain_type = seperated_3.chain_type.apply(lambda x: list(x)[0])
chain_type_result = pd.concat([seperated_1, seperated_2, seperated_3]).reset_index(drop=True)
label_chains_full


#%%
'''
# 去掉标签关系结果中本已属于同一链条的数据
label_chains_link = label_chains_full[["node_code", "root_code"]] \
    .apply(lambda x: x[0] + "-" + x[1] if int(x[0])>=int(x[1]) else x[1]+ "-" + x[0], axis=1).reset_index()
label_chains_link.columns = ["mark", "node_root_link"]
label_chains_link.mark = label_chains_link.mark.apply(lambda x: 1)
print(len(label_chains_link))
# len(statistic_result_py_df.merge(label_chains_link, how='inner', left_on='tag_link', right_on='node_root_link'))
tag_relation_with_link = statistic_result_py_df.merge(label_chains_link, how='left', left_on='tag_link', right_on='node_root_link')
tag_relation_value = tag_relation_with_link[tag_relation_with_link.mark != 1.0][["tag1", "tag2", "percentage"]]

#%%
# 相对关联度（与本标签之间的绝对强度占所有相关标签强度总和之比例）
tag_relation_value = statistic_result_py_df[["tag1", "tag2", "percentage"]]
tag_tag = tag_relation_value[["tag1", "tag2", "percentage"]]
tag_tag_reverse = tag_tag.copy()
tag_tag_reverse.columns = ["tag2", "tag1", "percentage"]
tag_tag_reverse
#%%
link_bidirect = pd.concat([tag_tag, tag_tag_reverse])
grouped_link = link_bidirect.groupby(["tag1", "tag2"]).agg({"percentage": "sum"})
relative_link = grouped_link.groupby(level=0).apply(lambda x: x/float(x.sum())).reset_index()
relative_link.percentage = relative_link.percentage.apply(lambda x: -np.log2(min(0.000001 + x, 1)))
target = relative_link.percentage.values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(target)
relative_link.percentage = scaler.transform(target)
relative_link.columns = ["tag1", "tag2", "distance"]
relative_link.describe()
'''
#%%
abs_link = statistic_result_py_df[["tag1", "tag2", "percentage"]].copy()
abs_link.percentage = abs_link.percentage.apply(lambda x: -np.log2(min(0.000001 + x, 1)))
target = abs_link.percentage.values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(target)
abs_link.percentage = scaler.transform(target)
abs_link.columns = ["tag1", "tag2", "distance"]
abs_link.describe()
#%%
# 生成graph
edges = abs_link.drop_duplicates()
edges.columns = ["source", "target", "weight"]
edges

#%%
G = nx.Graph(edges)
#%% 
a = list(nx.shortest_simple_paths(G,"000", "029", weight="weight"))
a


#%%
sc.stop()
