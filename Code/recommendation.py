# -*- coding: utf-8 -*-

import pickle
import pandas as pd
import numpy as np
import configparser
import multiprocessing as mp
import ast
from functools import reduce
from itertools import product
from Code import pipline_new, data_generator
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler



comp_infos, tag_tag, comp_id_name_dict, tag_dict = pipline_new.data_loader()


def cal_tag_cartesian(tag_set1, tag_set2, tag_link_filter, value_dict, tag_tag_weights):
    if tag_set1 == 0 or tag_set2 == 0:
        return 0
    else:
        pair_list = list(product(tag_set1, tag_set2))
        value_list = [value_dict.get(t[0] + "-" + t[1], (0.0, 0)) for t in pair_list]
        value_sum = sum([v[0]*tag_tag_weights[v[1]] for v in value_list if v[0] >= tag_link_filter])
        return value_sum
    
def cal_tags_link(tag_set1, tag_set2, tag_link_filter, value_dict, tag_tag_weight):
    num_1 = len(tag_set1)
    num_2 = len(tag_set2)
    jaccard_sim = data_generator.final_count(tag_set1, tag_set2)
    
    punish_coef = 1/np.sqrt((1 + num1) * (1 + num2))
    udf_sim = punish_coef * cal_tag_cartesian(tag_set1, tag_set2, tag_link_filter, value_dict, tag_tag_weights)
    
    return (jaccard_sim, udf_sim)

def cal_company_dis(target_comp_info, part, tag_link_filter, value_dict, tag_tag_weights, merge_weights):
    # print("start")
    value_list = list(part.comp_tag_list.apply(lambda x: cal_tags_link(target_comp_info, x, tag_link_filter, value_dict, tag_tag_weights)))
    part["value_list"] = value_list
    # print("end")
    return part


def multi_process_rank(comp_name, comp_info="", tag_link_filters=0.2, value_dict = tag_tag, tag_tag_weights=(0.5, 0.3, 0.2), merge_weights=(0.6, 0.4), response_num=None, , process_num=8):
    if comp_info == "":
        comp_id_name_dict_reverse = dict(zip(comp_id_name_dict.values(), comp_id_name_dict.keys()))
        comp_id = comp_id_name_dict_reverse.get(comp_name)
        target_comp_info = list(comp_infos[comp_infos.comp_id == comp_id].comp_tag_list)[0]
    # 只提供标签，此处待修改
    else:
        data_dict = ast.literal_eval(comp_info)
        target_comp_info = {k: set([tag_dict.get(t) for t in v]) for k, v in data_dict.items()}
        comp_id = ""
    
    print("start!")
    # 如果目标公司具备概念标签，则概念-非概念关系值的权重保留较高，否则提高非概念标签之间值的权重
    '''
    if target_comp_info.get("ctags") == None:
        tag_tag_weights = (tag_tag_weights[0], tag_tag_weights[2], tag_tag_weights[1])
    '''
    result_list = []
    split_comp_infos = np.array_split(comp_infos, process_num)
    pool = mp.Pool()
    for i in range(0, process_num):
        result_list.append(pool.apply_async(cal_company_dis, (target_comp_info, split_comp_infos[i], tag_link_filter, value_dict, tag_tag_weights, merge_weights,)))
    pool.close()
    pool.join()
    result_merged = pd.concat([r.get() for r in result_list])
    print("end!")
    result_merged.drop_duplicates(subset=["comp_id"], inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 100))
    to_transform = np.array(list(result_merged.value_list))
    scaler.fit(to_transform)
    result_merged["sim_value"] = (scaler.transform(to_transform) *  merge_weights).sum(axis=1)
    result_merged = result_merged[result_merged.comp_id != comp_id]
    result_merged.reset_index(drop=True, inplace=True)

    if response_num == None:
        response_num = len(result_merged)
    result_sorted = result_merged.sort_values(by="sim_value", ascending=False)[:response_num].copy()
    result_sorted['comp_name'] = result_sorted.comp_id.apply(lambda x: comp_id_name_dict.get(x))
    result_sorted.reset_index(drop=True, inplace=True)
    return result_sorted[["comp_id", "comp_name", "sim_value"]]
    
