# coding: utf-8

import pandas as pd
import numpy as np
import pyspark
import operator
import datetime
import sys
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
    return len(l1.intersection(l2))/len(l1.union(l2))
    

file_name = "company_tag_data_non_concept"
data_raw = pd.read_csv("../Data/Input/Tag_graph/" + file_name, sep='\t', dtype={"comp_id": str, "comp_full_name": str, "key_word": str})[["comp_id", "comp_full_name", "key_word"]]
data_raw.dropna(subset=["comp_id", "key_word"], inplace=True)
data_raw = data_raw[data_raw.key_word != ""]

tree_tuples = data_raw.apply(lambda x: [(x[0], x[1], t) for t in x[2].split(",") if t != ""], axis=1)
flatted = [y for x in tree_tuples for y in x]
non_concept = pd.DataFrame(flatted, columns=["comp_id", "comp_full_name", "tag"]).drop_duplicates()
id_dict = non_concept["comp_id"].drop_duplicates().reset_index(drop=True)
id_dict = id_dict.reset_index()
id_dict.rename(index=str, columns={"index": "int_id"}, inplace=True)
print(id_dict)
non_concept = non_concept.merge(id_dict, how="left", left_on="comp_id", right_on="comp_id").drop(["comp_id"], axis=1)
print(non_concept)
# sys.exit(0)
non_concept["count_comps"] = non_concept.tag
comps_per_nc = non_concept.groupby("tag").agg({"int_id": lambda x: set(x), "count_comps":"count"}).reset_index()
comps_per_nc_part = comps_per_nc[comps_per_nc.count_comps >= 50][["tag", "int_id"]].reset_index(drop=True)
comps_per_nc_part["key"] = 1

print("start full merge")
full = comps_per_nc_part.merge(comps_per_nc_part, on="key")
# full["code_link"] = full[["tag_code_x", "tag_code_y"]].apply(x[0] + "-" + x[1] if int(x[0])>=int(x[1]) else x[1]+ "-" + x[0], axis=1)
# ordered = pd.DataFrame(set(full[["tag_code_x", "tag_code_y", "comp_id_x", "comp_id_y"]].apply(lambda x: (x[0], x[1], x[2], x[3]) if int(x[0])>=int(x[1]) else (x[1], x[0], x[3], x[2]), axis=1)))
# ordered.columns = ["tag_x", "tag_y", "comp_id_x", "comp_id_y"]
# del(full)
# ordered.drop_duplicates(inplace=True)
ordered = full
record_len = len(ordered)
interval_size = record_len // 100
print("full lenght is %d." % record_len)
print("interval size is %d" % interval_size)
i = 0
while interval_size*i < record_len:
	start_time = datetime.datetime.now()
	print("### start part %d at %s ###" % (i, start_time.strftime('%H:%M:%S')))
	tmp = ordered[interval_size*i:min(interval_size*(i+1), record_len)]
	tmp["link_value"] = tmp[["int_id_x", "int_id_y"]].apply(lambda x: final_count(x[0], x[1]), axis=1)
	print("calculation done")
	result_part = tmp[(tmp.tag_x != tmp.tag_y) & (tmp.link_value > 0)][["tag_x", "tag_y", "link_value"]]
	result_part.to_csv("../Data/Output/Tag_graph/part_result_%d.relations" % i, index=False, header=None)
	end_time = datetime.datetime.now()
	print("### Part %d finished at %s (time used: %.3f seconds) ###" % (i, end_time.strftime('%H:%M:%S'), (end_time - start_time).total_seconds()))
	i += 1
	del(tmp)
	del(result_part)

print("Process finished successfully!")
