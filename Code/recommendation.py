# -*- coding: utf-8 -*-

import pickle
import pandas as pd
import numpy as np
import configparser
import multiprocessing as mp
import ast
from functools import reduce
from itertools import product
from Code import pipline_new
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from scipy.sparse import coo_matrix



comp_infos, tag_tag, comp_id_name, tag_dict, comp_tags_all_sparse, tag_tag_sparse = pipline_new.data_loader()
# comp_infos, comp_tags_all_sparse, tag_tag_sparse, comp_id_name , tag= pipline_new.data_loader_sparse()

tag_dict_reverse = dict(zip(tag_dict.values(), tag_dict.keys()))

def final_count(l1, l2):
    len1 = len(l1)
    len2 = len(l2)
    intersect = len(l1.intersection(l2))
    if intersect == 0:
        return 0
    else:
        return intersect/(len1 + len2 - intersect)

def cal_tag_cartesian(tag_set1, tag_set2, tag_link_filter, value_dict, tag_tag_weights):
    if len(tag_set1) == 0 or len(tag_set2) == 0:
        return 0
    else:
        pair_list = list(product(tag_set1, tag_set2))
        value_list = [value_dict.get(t[0] + "-" + t[1], (0.0, 0)) for t in pair_list]
        value_sum = sum([v[0]*tag_tag_weights[v[1]] for v in value_list if v[0] >= tag_link_filter])
        return value_sum
    
def cal_tags_link(tag_set1, tag_set2, tag_link_filter, value_dict, tag_tag_weights):
    jaccard_sim = final_count(tag_set1, tag_set2)
    udf_sim = cal_tag_cartesian(tag_set1, tag_set2, tag_link_filter, value_dict, tag_tag_weights)
    
    return (jaccard_sim, udf_sim)

def cal_company_dis(target_comp_info, part, tag_link_filter, value_dict, tag_tag_weights, merge_weights):
    # print("start")
    value_list = list(part.comp_tag_list.apply(lambda x: cal_tags_link(target_comp_info, x, tag_link_filter, value_dict, tag_tag_weights)))
    part["value_list"] = value_list
    # print("end")
    return part


def multi_process_rank(comp_name, comp_info="", tag_link_filter=0.2, value_dict = tag_tag, tag_tag_weights=(0.5, 0.3, 0.2), merge_weights=(0.6, 0.4), response_num=None, process_num=8):
    if comp_info == "":
        comp_id = list(comp_id_name[comp_id_name.name == comp_name].point_id)[0]
        target_comp_info = list(comp_infos[comp_infos.comp_id == comp_id].comp_tag_list)[0]
        target_tags =  ",".join([tag_dict_reverse.get(x) for x in target_comp_info])
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
    split_comp_infos_index = np.array_split(comp_infos.index, process_num)
    pool = mp.Pool()
    for i in range(0, process_num):
        result_list.append(pool.apply_async(cal_company_dis, (target_comp_info, comp_infos.iloc[split_comp_infos_index[i].tolist()], tag_link_filter, value_dict, tag_tag_weights, merge_weights,)))
    pool.close()
    pool.join()
    result_merged = pd.concat([r.get() for r in result_list])
    print("end!")
    result_merged.drop_duplicates(subset=["comp_id"], inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 10))
    to_transform = np.array(list(result_merged.value_list))
    # to_transform = result_merged.value_list.values.reshape(-1, 1)
    scaler.fit(to_transform)
    
    result_merged["sim_value"] = scaler.transform(to_transform).cumprod(axis=1)[:, 1:2]
    #(scaler.transform(to_transform) * merge_weights).sum(axis=1)
    result_merged = result_merged[result_merged.comp_id != comp_id]
    result_merged.reset_index(drop=True, inplace=True)

    if response_num == None:
        response_num = len(result_merged)
    result_sorted = result_merged.sort_values(by="sim_value", ascending=False)[:response_num].copy()
    result_sorted = result_sorted.merge(comp_id_name[["point_id", "name"]], how="left", left_on="comp_id", right_on="point_id")
    result_sorted["tag_list"] = result_sorted.comp_tag_list.apply(lambda x: ",".join([tag_dict_reverse.get(w) for w in x]))
    result_sorted.reset_index(drop=True, inplace=True)
    return (target_tags, result_sorted[["comp_id", "name", "tag_list", "sim_value"]])
    

def sparse_cal(comp_name, metric='cosine', comp_info="", sort_key="sim_value_prod", merge_weights=(0.6, 0.4), response_num=None):
    comp_id_list = list(comp_id_name[comp_id_name.name == comp_name].point_id)
    if len(comp_id_list) == 0:
        return -1
    comp_id = comp_id_list[0]
    if comp_info == "":    
        target_info = comp_infos.loc[comp_infos.comp_id == comp_id].comp_tag_list.tolist()[0]
        target_index = comp_infos.loc[comp_infos.comp_id == comp_id].index.tolist()[0]
        target_tags =  ",".join([tag_dict_reverse.get(x) for x in target_info])
        target_sparse_vector = comp_tags_all_sparse.getrow(target_index)
    else:
        tag_str, value_str = comp_info.split("#")
        tag_list = [tag_dict.get(t) for t in tag_str.split(",")]
        value_list = [float(v) for v in value_str.split(",")]
        if len(tag_list) != len(value_list):
            print("Numbers of labels and weights do not match!")
            return -1
        # print(tag_list)
        # tag_value_list = [(tag_list[i], value_list[i]) for i in range(0, len(tag_list)) if tag_list[i] != None]
        tag_index = dict(zip(tag_dict.values(), range(0, len(tag_dict))))
        col_index = [tag_index.get(w) for w in tag_list if w != None]
        value_list_left = [value_list[i] for i in range(0, len(tag_list)) if tag_list[i] != None]
        length = len(tag_index)
        target_tags = tag_str
        target_sparse_vector = coo_matrix((np.array(value_list_left)/len(value_list_left), (np.zeros(len(value_list_left), dtype=int), col_index)), shape=(1, length))
        
    # cosine similarity
    if metric == "cosine":
        sim = 1 - pairwise_distances(comp_tags_all_sparse, target_sparse_vector, metric=metric)
    elif metric == "euclidean":
        sim = 1/(1 + (pairwise_distances(comp_tags_all_sparse, target_sparse_vector, metric=metric)))
    else:
        print("Metric: %s is not supported" % metric)
        return -1
    # udf similarity
    udf_sim = comp_tags_all_sparse.dot(tag_tag_sparse.dot(target_sparse_vector.T)).toarray()
    
    comp_infos["value_list"] = np.concatenate((sim, udf_sim), axis=1).tolist()
    scaler1 = MinMaxScaler(feature_range=(0, 100))
    scaler2 = MinMaxScaler(feature_range=(0, 100))
    to_transform = np.array(list(comp_infos.value_list))
    # to_transform = result_merged.value_list.values.reshape(-1, 1)
    scaled = scaler1.fit_transform(to_transform)
    comp_infos["sim_value_prod"] = sim * udf_sim
    comp_infos.sim_value_prod = scaler2.fit_transform(comp_infos.sim_value_prod.values.reshape(-1, 1))
    comp_infos["sim_value_sum"] = (scaled * merge_weights).sum(axis=1)
    if response_num == None:
        response_num = len(comp_infos)
    result_sorted = comp_infos[["comp_id", "comp_tag_list", "value_list", "sim_value_prod", "sim_value_sum"]] \
        .sort_values(by=sort_key, ascending=False)[: response_num + 1]
    result_sorted = result_sorted.merge(comp_id_name[["point_id", "name"]], how="left", left_on="comp_id", right_on="point_id")
    '''
    #########################################
    注意！！！！！！！！！！！！此后这里需要改为w[0]
    #########################################
    '''
    result_sorted["tag_list"] = result_sorted.comp_tag_list.apply(lambda x: ",".join([tag_dict_reverse.get(w) for w in x]))
    result_sorted.reset_index(drop=True, inplace=True)
    return (comp_id, target_tags, result_sorted[["comp_id", "name", "tag_list", "value_list", sort_key]])
    
    
def matrix_similarities(target_comps, search_pool):
    target_ids = comp_id_name[comp_id_name.name.isin(target_comps)].point_id.tolist()
    search_pool_ids = comp_id_name[comp_id_name.name.isin(search_pool)].point_id.tolist()
    
    target_indexes = comp_infos[comp_infos.comp_id.isin(target_ids)].index.tolist()
    search_pool_indexes = comp_infos[comp_infos.comp_id.isin(search_pool_ids)].index.tolist()
    # print(len(target_ids), len(search_pool_ids), len(target_indexes), len(search_pool_indexes))
    
    column_name = comp_id_name[comp_id_name.point_id.isin(comp_infos.loc[target_indexes].comp_id)].name.tolist()
    index_name = comp_id_name[comp_id_name.point_id.isin(comp_infos.loc[search_pool_indexes].comp_id)].name.tolist()

    csr_matrix = comp_tags_all_sparse.tocsr()
    target_matrix = csr_matrix[target_indexes, :]
    search_pool_matrix = csr_matrix[search_pool_indexes]
    udf_sim = search_pool_matrix.dot(tag_tag_sparse.dot(target_matrix.T)).toarray()
    cosine_sim = 1 - pairwise_distances(search_pool_matrix, target_matrix, metric="cosine")

    sim_prod = cosine_sim * udf_sim
    result = pd.DataFrame(sim_prod, index=index_name, columns=column_name)
    return result
        
        
        