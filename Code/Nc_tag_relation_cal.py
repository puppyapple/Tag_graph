# coding: utf-8

import pandas as pd
import numpy as np
import pyspark
import operator
from pyspark import SparkContext 
from pyspark.sql import SQLContext 
from collections import Counter
from itertools import groupby
from functools import reduce
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

def union_count(l1, l2):
    return len(set(set(l1).union(set(l2))))

def intersection_count(l1, l2):
    return len(set(set(l1).intersection(set(l2))))

def final_count(l1, l2):
    ic = intersection_count(l1, l2)
    uc = union_count(l1, l2)
    return(ic, uc, ic/uc)


file_name = "company_tag_data_non_concept"
data_raw = pd.read_csv("../Data/Input/" + file_name, sep='\t', dtype={"comp_id": str, "comp_full_name": str, "key_word": str})[["comp_id", "comp_full_name", "key_word"]]
data_raw.dropna(subset=["comp_id", "key_word"], inplace=True)
data_raw = data_raw[data_raw.key_word != ""]

tree_tuples = data_raw.apply(lambda x: [(x[0], x[1], t) for t in x[2].split(",") if t != ""], axis=1)
flatted = [y for x in tree_tuples for y in x]
non_concept = pd.DataFrame(flatted, columns=["comp_id", "comp_full_name", "tag"]).drop_duplicates()
non_concept["count_comps"] = non_concept.tag
comps_per_nc = non_concept.groupby("tag").agg({"comp_id": lambda x: list(x), "count_comps":"count"}).reset_index()


comps_per_nc_part = comps_per_nc[comps_per_nc.count_comps >= 50][["tag", "comp_id"]]
comps_per_nc_part["key"] = 1
full = comps_per_nc_part.merge(comps_per_nc_part, on="key")
full["relation"] = full[["comp_id_x", "comp_id_y"]].apply(lambda x: final_count(x[0], x[1]), axis=1)

result_raw = full[full.tag_x != full.tag_y][["tag_x", "tag_y", "relation"]].copy()
result_raw["intersection"] = result_raw.relation.apply(lambda x: x[0])
result_raw["union"] = result_raw.relation.apply(lambda x: x[1])
result_raw["link_value"] = result_raw.relation.apply(lambda x: x[2])

result = result_raw[result_raw.intersection > 0].drop("relation", axis=1)
print(len(result))
result.link_value = result.link_value.apply(lambda x: np.log2(min(0.000001 + x, 1)))
target = result.link_value.values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0.001, 1))
scaler.fit(target)
result.link_value = scaler.transform(target)
result.to_csv("../Data/Output/nc_tag_relation_raw", index=False)

print("Process finished successfully!")
