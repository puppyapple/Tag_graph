# -*- coding: utf-8 -*-
#%%
import pandas as pd
import numpy as np
import pyspark
import operator
import matplotlib.pyplot as plt
from pyspark import SparkContext
from pyspark.sql import SQLContext
from collections import Counter
from itertools import groupby
from functools import reduce
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

#%%
file_name = "company_tag_data_raw"

data_raw = pd.read_csv("../Data/Input/Tag_graph/" + file_name, sep='\t', dtype={"comp_id":str})[["comp_id", "comp_full_name", "label_name", "classify_id", "label_type", "label_type_num", "src_tags"]]
data_raw.comp_full_name = data_raw.comp_full_name.apply(lambda x: x.strip().replace("(","（").replace(")","）"))
non_concept_raw = data_raw[data_raw.classify_id == 4][['label_name', 'comp_id']].drop_duplicates()
non_concept_raw.columns = ['non_concept_label', 'comp_id']

sc = SparkContext.getOrCreate()
sqlContext=SQLContext(sc)

non_concept_df = sqlContext.createDataFrame(non_concept_raw)
non_concept_gb_df = sqlContext.createDataFrame(non_concept_df.rdd.flatMap(lambda x: map(lambda y: (y, x[1]), x[0].split(','))), schema=['non_concept_label', 'comp_id'])
non_concept = non_concept_gb_df.toPandas()

#%%
def union_count(l1, l2):
    return len(set(set(l1).union(set(l2))))

def intersection_count(l1, l2):
    return len(set(set(l1).intersection(set(l2))))

def final_count(l1, l2):
    ic = intersection_count(l1, l2)
    uc = union_count(l1, l2)
    return(ic, uc, ic/uc)

def final_count2(l1, l2):
    ic = intersection_count(l1, l2)
    uc = union_count(l1, l2)
    return [ic, uc, ic/uc]

#%%
non_concept["count_comps"] = non_concept.non_concept_label
comps_per_nc = non_concept.groupby("non_concept_label").agg({"comp_id": lambda x: list(x), "count_comps":"count"}).reset_index()

#%%
comps_per_nc_df = sqlContext.createDataFrame(comps_per_nc[comps_per_nc.count_comps >= 5][["non_concept_label", "comp_id"]])
comps_per_nc_df2 = comps_per_nc_df.withColumnRenamed("non_concept_label" ,"non_concept_label2")     .withColumnRenamed("comp_id" ,"comp_id2")
comps_per_nc_df.show(10)

all_relation = comps_per_nc_df.crossJoin(comps_per_nc_df2).filter("non_concept_label != non_concept_label2")
statistic_result_rdd = all_relation.rdd.map(lambda x: (x[0], x[2]) + final_count(x[1], x[3])).distinct()
statistic_result_df = sqlContext.createDataFrame(statistic_result_rdd, schema=["tag1", "tag2", "intersection", "union", "percentage"]).filter("intersection != 0")
statistic_result_py_df = statistic_result_df.toPandas()
statistic_result_df.show()

#%%
comps_per_nc_part = comps_per_nc[comps_per_nc.count_comps >= 5][["non_concept_label", "comp_id"]]
comps_per_nc_part["key"] = 1
a = comps_per_nc_part.merge(comps_per_nc_part, on="key")

#%%
a["relation"] = a[["comp_id_x", "comp_id_y"]].apply(lambda x: final_count(x[0], x[1]), axis=1)
 
result_raw = a[a.non_concept_label_x != a.non_concept_label_y][["non_concept_label_x", "non_concept_label_y", "relation"]].copy()
result_raw["intersection"] = result_raw.relation.apply(lambda x: x[0])
result_raw["union"] = result_raw.relation.apply(lambda x: x[1])
result_raw["link_value"] = result_raw.relation.apply(lambda x: x[2])

#%%
result = result_raw[result_raw.intersection > 0].drop("relation", axis=1)
print(len(result))
result.link_value = result.link_value.apply(lambda x: np.log2(min(0.000001 + x, 1)))
target = result.link_value.values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0.001, 1))
scaler.fit(target)
result.link_value = scaler.transform(target)
result.to_csv("../Data/Output/Tag_graph/non_concept_tag.relations", index=False)

#%%
result[result.non_concept_label_y == "皮革"].sort_values(by="link_value")