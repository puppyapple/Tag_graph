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
# 公司标签数据
header_dict = {
    "point": ["point_id", "name", "property", "point_type"],
    "relation": ["src_id", "target_id", "rel_value", "rel_type"]
}

#%% 
# ******非概念部分******
file_name_nc = "company_tag_data_non_concept"
data_raw = pd.read_csv("../Data/Input/" + file_name_nc, sep='\t', dtype={"comp_id": str, "comp_full_name": str, "key_word": str})[["comp_id", "comp_full_name", "key_word"]]
data_raw.dropna(subset=["comp_id", "key_word"], inplace=True)
data_raw = data_raw[data_raw.key_word != ""]

tree_tuples = data_raw.apply(lambda x: [(x[0], x[1], t) for t in x[2].split(",") if t != ""], axis=1)
flatted = [y for x in tree_tuples for y in x]
non_concept = pd.DataFrame(flatted, columns=["comp_id", "comp_full_name", "tag"]).drop_duplicates()
non_concept["count_comps"] = non_concept.tag
comps_per_nc = non_concept.groupby("tag").agg({"comp_id": lambda x: list(x), "count_comps":"count"}).reset_index()
comps_per_nc_part = comps_per_nc[comps_per_nc.count_comps >= 50][["tag", "comp_id"]]
comps_per_nc_part

#%%
# 非概念标签字典
comps_per_nc_part.reset_index(drop=True, inplace=True)
tag_code_dict_nc = comps_per_nc_part.tag.reset_index()
tag_code_dict_nc.columns = ["tag_code", "label_name"]
length = len(str(len(tag_code_dict_nc)))
tag_code_dict_nc.tag_code = tag_code_dict_nc.tag_code.apply(lambda x: str(x).zfill(length))
tag_code_dict_nc

#%% 
# ******概念部分******
# 读取数据
sc = SparkContext.getOrCreate()
sqlContext=SQLContext(sc)
file_name_c = "company_tag_data_concept"
data_raw_c = pd.read_csv("../Data/Input/" + file_name_c, sep='\t', dtype={"comp_id":str})
cols = ["comp_id", "comp_full_name", "label_name", "classify_id", "label_type", "label_type_num", "src_tags"]
data_raw_c = data_raw_c[cols]
concept_tags = data_raw_c[data_raw_c.classify_id != 4].reset_index(drop=True)
concept_tags.label_name = concept_tags[["label_name", "label_type_num", "src_tags", "label_type"]] \
    .apply(lambda x: x[2].split("#")[x[1]-1].split("-")[max(x[3]-2, 0)] + ":" + x[0], axis=1)
#%%
# 概念关系表字典
level_data_raw_c = pd.read_csv("../Data/Input/label_code_relation", sep='\t', dtype={"label_root_id":str, "label_note_id":str})
tag_code_dict_c = pd.concat([level_data_raw_c.label_note_name, level_data_raw_c.label_root_name]).drop_duplicates().reset_index(drop=True)
tag_code_dict_c.name = "label_name"
tag_code_dict_c = tag_code_dict_c.reset_index()
tag_code_dict_c.rename(index=str, columns={"index": "tag_code"}, inplace=True)
length = len(str(len(tag_code_dict_c)))
tag_code_dict_c.tag_code = tag_code_dict_c.tag_code.apply(lambda x: str(x).zfill(length))

#%%
# 按标签聚合公司
comps_by_tag = concept_tags[["label_name", "comp_id"]].groupby("label_name").agg(pinjie).reset_index()
comps_by_tag["label_name_father"] = comps_by_tag.label_name.apply(lambda x: x.split(":")[0])
comps_by_tag["label_name"] = comps_by_tag.label_name.apply(lambda x: x.split(":")[1])
comps_by_tag_rough = comps_by_tag.groupby("label_name").agg(pinjie).reset_index()[["label_name", "comp_id"]]
concept_tags

#%%
# 为标签分组表赋上标签代码
comps_by_tag_rough_with_code = comps_by_tag_rough.merge(tag_code_dict_c, how='left', left_on='label_name', right_on='label_name') \
    .dropna(how='any')
tag_code_dict_c_2 = tag_code_dict_c.copy()
tag_code_dict_c_2.columns = ["tag_code_father", "label_name_father"]
comps_by_tag_with_code = comps_by_tag.merge(tag_code_dict_c, how='left', left_on='label_name', right_on='label_name') \
    .merge(tag_code_dict_c_2, how='left', left_on='label_name_father', right_on='label_name_father') \
    .dropna(how='any')
comps_by_tag_with_code

#%%
# 概念标签之间的层级关系（tag1 belongs to tag2）
label_chains_raw = level_data_raw_c.reset_index(drop=True) \
    .rename(index=str, columns={"label_note_name":"label_node_name", "label_type_note":"label_type_node"}, inplace=False)
tag_code_root = tag_code_dict_c.rename(index=str, columns={"tag_code":"root_code", "label_name":"root_name"}, inplace=False)
tag_code_node = tag_code_dict_c.rename(index=str, columns={"tag_code":"node_code", "label_name":"node_name"}, inplace=False)
label_chains_full = label_chains_raw.merge(tag_code_node, how='left', left_on='label_node_name', right_on='node_name') \
    .merge(tag_code_root, how='left', left_on='label_root_name', right_on='root_name')[["node_code", "root_code", "label_type_node", "label_type_root"]]
label_chains = label_chains_full[level_data_raw_c.label_type_root-level_data_raw_c.label_type_note==-1]

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
tag_chains = label_chains[["node_code", "root_code", "proportion"]].copy()
tag_chains["rel_type"] = "node_of"
tag_chains.columns = header_dict["relation"]
tag_chains
#%%
# 非层级概念标签关系
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
label_chains_link = label_chains_full[["node_code", "root_code"]] \
    .apply(lambda x: x[0] + "-" + x[1] if int(x[0])>=int(x[1]) else x[1]+ "-" + x[0], axis=1).reset_index()
label_chains_link.columns = ["mark", "node_root_link"]
label_chains_link.mark = label_chains_link.mark.apply(lambda x: 1)
print(len(label_chains_link))
# len(statistic_result_py_df.merge(label_chains_link, how='inner', left_on='tag_link', right_on='node_root_link'))
tag_relation_with_link = statistic_result_py_df.merge(label_chains_link, how='left', left_on='tag_link', right_on='node_root_link')
tag_relation_value_c = tag_relation_with_link[tag_relation_with_link.mark != 1.0][["tag1", "tag2", "percentage"]]
tag_relation_value_c.percentage = tag_relation_value_c.percentage.apply(lambda x: np.log2(min(0.000001 + x, 1)))
target = tag_relation_value_c.percentage.values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0.001, 1))
scaler.fit(target)
tag_relation_value_c.percentage = scaler.transform(target)
tag_relation_value_c["rel_type"] = "ctag_ctag"
tag_relation_value_c.columns = header_dict["relation"]
tag_relation_value_c
#%% 
# 公司和标签关系(company belongs to tag_x -> ... -> tag_1)
concept_tags.label_name = concept_tags.label_name.apply(lambda x: x.split(":")[1])
concept_tags_with_code = concept_tags.merge(tag_code_dict_c, how='left', left_on='label_name', right_on='label_name') \
    .dropna(how='any')
company_tag_relations_c = concept_tags_with_code.groupby(["comp_id", "label_type_num"]).apply(lambda x: x[x.label_type == x.label_type.max()])
company_tag_relations_c = company_tag_relations_c[["comp_id", "tag_code"]].drop_duplicates().reset_index(drop=True)

company_tag_relations_c
#%%
company_tag_relations_nc = non_concept.merge(tag_code_dict_nc, how="left", left_on="tag", right_on="label_name").dropna(how="any")
company_tag_relations_nc = company_tag_relations_nc[["comp_id", "tag_code"]].drop_duplicates()
company_tag_relations_nc
company_tag_relations = pd.concat([company_tag_relations_c, company_tag_relations_nc])
company_tag_relations["rel_value"] = 0
company_tag_relations["rel_type"] = "company_tag"
company_tag_relations.columns = header_dict["relation"]
company_tag_relations

#%%
# 非概念标签关系
link_value_raw = pd.read_csv("../Data/Input/all_links_raw.csv", header=None)
#%%
link_value_raw.columns = ["tag1", "tag2", "link_value"]
tag_code_dict_nc_1 = tag_code_dict_nc.copy().rename(index=str, columns={"tag_code": "code1", "label_name": "tag1"})
tag_code_dict_nc_2 = tag_code_dict_nc.copy().rename(index=str, columns={"tag_code": "code2", "label_name": "tag2"})
tag_relation_value_nc_raw = link_value_raw.merge(tag_code_dict_nc_1, how="left", left_on="tag1", right_on="tag1") \
    .merge(tag_code_dict_nc_2, how="left", left_on="tag2", right_on="tag2")
#%%
tag_relation_value_nc_raw.dropna(how="any", inplace=True)
tag_relation_value_nc_raw.drop(["tag1", "tag2"], axis=1, inplace=True)
tag_relation_value_nc_raw["code_link"] = tag_relation_value_nc_raw[["code1", "code2"]] \
    .apply(lambda x: x[0] + "-" + x[1] if int(x[0])>=int(x[1]) else x[1]+ "-" + x[0], axis=1)
tag_relation_value_nc_raw
nc_tag_link_no_duplicates = tag_relation_value_nc_raw.drop(["code1", "code2"], axis=1).drop_duplicates()
nc_tag_link_no_duplicates["code1"] = nc_tag_link_no_duplicates.code_link.apply(lambda x: x.split("-")[0])
nc_tag_link_no_duplicates["code2"] = nc_tag_link_no_duplicates.code_link.apply(lambda x: x.split("-")[1])
nc_tag_link_no_duplicates

#%%
tag_relation_value_nc = nc_tag_link_no_duplicates[["code1", "code2", "link_value"]].copy()
tag_relation_value_nc.link_value = tag_relation_value_nc.link_value.apply(lambda x: np.log2(min(0.000001 + x, 1)))
target = tag_relation_value_nc.link_value.values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0.001, 1))
scaler.fit(target)
tag_relation_value_nc.link_value = scaler.transform(target)
tag_relation_value_nc["rel_type"] = "nctag_nctag"
tag_relation_value_nc.columns = header_dict["relation"]
tag_relation_value_nc


#%%
# 节点数据
# 公司
companies_c = concept_tags_with_code[["comp_id", "comp_full_name"]].drop_duplicates()
companies_c.comp_full_name = companies_c.comp_full_name.apply(lambda x: x.strip().replace("(","（").replace(")","）"))
companies_c = companies_c.groupby("comp_id").apply(lambda x: x[x.comp_full_name==x.comp_full_name.max()]).drop_duplicates().reset_index(drop=True)
companies_c
#%%
companies_c["property"] = ""
companies_c["point_type"] = "company"
companies_c.columns = header_dict["point"]
companies_nc = non_concept.merge(comps_per_nc_part[["tag"]], how="left", left_on="tag", right_on="tag").dropna(how="any")[["comp_id", "comp_full_name"]] \
    .drop_duplicates()
companies_nc["property"] = ""
companies_nc["point_type"] = "company"
companies_nc.columns = header_dict["point"]
#%%
companies = pd.concat([companies_c, companies_nc]).drop_duplicates()
#%%
# 标签
tags_c = concept_tags_with_code[["tag_code", "label_name"]].drop_duplicates().reset_index(drop=True)
tags_c["property"] = "concept"
tags_c["point_type"] = "tag"
tags_c.columns = header_dict["point"]

tags_nc = tag_code_dict_nc.copy()
tags_nc["property"] = "non_concept"
tags_nc["point_type"] = "tag"
tags_nc.columns = header_dict["point"]

tags = pd.concat([tags_c, tags_nc]).drop_duplicates()
tags.columns = header_dict["point"]
tags

#%%
# 全部点集合
points = pd.concat([companies, tags]).drop_duplicates().reset_index(drop=True)
points.columns = header_dict["point"]
points.reset_index(inplace=True)
points.rename(index=str, columns={"index": "id"}, inplace=True)
points.id = points.id.apply(lambda x: x + 1)
points.to_csv("../Data/Full_data/all_points.csv", index=False, header=None)
print("Points saved!")
#%%
# 边数据整合
relations = pd.concat([tag_chains, tag_relation_value_c, tag_relation_value_nc, company_tag_relations]).drop_duplicates().reset_index(drop=True)
relations.columns = header_dict["relation"]
relations.reset_index(inplace=True)
relations.rename(index=str, columns={"index": "id"}, inplace=True)
relations.id = relations.id.apply(lambda x: x + 1)
relations.to_csv("../Data/Full_data/all_relations.csv", index=False, header=None)
print("Relations saved!")