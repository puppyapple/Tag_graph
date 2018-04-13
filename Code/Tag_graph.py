#%%
import pandas as pd
import numpy as np 
import sys
from pandas import Series
from functools import reduce

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
    return result

#%%
data_raw = pd.read_csv("../Data/company_tag_data_raw", sep='\t', dtype={"comp_id":str})
cols = ["comp_id", "comp_full_name", "label_name", "classify_id", "label_type", "label_type_num", "src_tags"]
data_raw = data_raw[cols]
concept_tags = data_raw[data_raw.classify_id!=4].reset_index(drop=True)
concept_tags["label_code"] = pd.Series(concept_tags.index).apply(lambda x: str(x))
#length = len(str(len(concept_tags)))
tags_by_comp = concept_tags[["comp_id","label_code"]].groupby("comp_id").agg(pinjie).reset_index()

#%%
tag_code_dict = concept_tags.comp_full_name.drop_duplicates().reset_index(drop=True)

tag_code_dict

#tag_code_dict


#%%
length = len(str(len(tag_code_dict)))
tags_by_comp["tag_couple"] = tags_by_comp.label_code.apply(lambda x: tag_couple(x.split(','), length))
type(tags_by_comp.tag_couple[0])