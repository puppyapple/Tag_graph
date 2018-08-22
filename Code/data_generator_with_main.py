# -*- coding: utf-8 -*-
#%%
import pandas as pd
import numpy as np
import uuid
import pickle
import os
import configparser
import pymysql
import datetime
import multiprocessing as mp
import gc
from tqdm import tqdm
from pyspark import SQLContext, SparkContext
from pyspark.sql import SparkSession
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine

config = configparser.ConfigParser()
config.read("../Data/Input/database_config/database.conf")
host = config['ASSESSMENT']['host']
user = config['ASSESSMENT']['user']
password = config['ASSESSMENT']['password']
database = config['ASSESSMENT']['database']
port = config['ASSESSMENT']['port']
charset = config['ASSESSMENT']['charset']
db = pymysql.connect(host=host, user=user, password=password, db=database, port=int(port), charset=charset)
config.read("../Data/Input/Tag_graph/filter.conf")
ctag_keep_list = config['FILTER']['filter_list'].split(",")

engine = create_engine("mysql+pymysql://%s:%s@%s:%s/%s?charset=utf8" % (user, password, host, port, database))

points_table = "wuzj_local_points"
relations_table = "wuzj_local_relations"
tag_comps_table = "wuzj_local_tag_comps"
label_chains_table = "wuzj_local_label_chains"

header_dict = {
    "point": ["point_id", "name", "property", "point_type"],
    "relation": ["src_id", "target_id", "rel_value", "rel_type"]
}


# 基本数据处理及生成
def comp_tag(new_result="company_tag_info_latest0703", old_result="company_online_tag_info_latest", label_code_relation="label_code_relation", keep_list=False, db=db):
    # 从库中读取数据
    print("###### Loading data ######")
    sql_new_result = "select comp_id, comp_full_name, label_name, classify_id, label_type, label_type_num, main_business_type from %s" % new_result
    sql_old_result = "select comp_id, comp_full_name, label_name, main_business_type as 1.0 from %s" % old_result
    sql_label_code_relation = "select * from %s" % label_code_relation
    label_chains_raw = pd.read_sql(sql_label_code_relation, con=db)
    data_raw_new = pd.read_sql(sql_new_result, con=db)
    data_raw_old_full = pd.read_sql(sql_old_result, con=db)
    data_raw_new.fillna("", inplace=True)
    data_raw_old_full.fillna("", inplace=True)
    print("Data loaded")
    # 生成公司id-name字典保存
    print("###### Processing company_points ######")
    company_points = pd.concat([data_raw_new[["comp_id", "comp_full_name"]], data_raw_old_full[["comp_id", "comp_full_name"]]]).drop_duplicates()
    company_points.comp_full_name = company_points.comp_full_name.apply(lambda x: x.strip())
    
    '''
    ###############################
    * 公司id-名称储存用作导入图数据库 *
    ###############################
    '''
    company_points.drop_duplicates(subset=["comp_id"], inplace=True)
    company_points.dropna(how="any", inplace=True)
    company_points["property"] = ""
    company_points["point_type"] = "company"
    company_points.columns = header_dict.get("point")
    company_points.to_csv("../Data/Output/Tag_graph/company_points.points", header=None, index=False)
    # print("###### Appending company_points to mysql ######")
    # company_points.to_sql(name=points_table, con=engine, if_exists="append", index=False)
    print("compnay_points saved")
    
    # 全部概念标签的列表
    ctag_full_list = set(label_chains_raw.label_note_name).union(set(label_chains_raw.label_root_name))
    
    if keep_list == True:
        keep_list = set(ctag_keep_list).union(label_chains_raw[label_chains_raw.label_root_name.isin(ctag_keep_list)].label_note_name)
    else:
        keep_list = ctag_full_list
    
    # 根据输入的公司概念和非概念标记源数据，分别得到完整的公司-概念标签、公司-非概念标签
    data_raw_new.dropna(subset=["comp_id", "label_name"], inplace=True)  
    
    # 过滤掉进行中的产业链话题等关联的公司
    comp_ctag_table_all_infos = data_raw_new[data_raw_new.classify_id != 4].reset_index(drop=True)
    comp_ctag_table_all_infos = comp_ctag_table_all_infos[comp_ctag_table_all_infos.label_name.isin(keep_list)]
    comp_ctag_table = comp_ctag_table_all_infos[["comp_id", "label_name", "main_business_type"]].reset_index(drop=True)
    comp_ctag_table["type"] = 0
    
    # 新系统下结果中的公司-非概念标签
    data_raw_nctag_p1 = data_raw_new[data_raw_new.classify_id == 4][["comp_id", "label_name", "main_business_type"]]

    # 读取旧版数据，只取其中的非概念标签（概念标签无法确定其层级和产业链（复用））
    data_raw_nctag_p2 = data_raw_old_full[data_raw_old_full.label_name != ""][["comp_id", "label_name", "main_business_type"]]
    data_raw_nctag_p2.dropna(subset=["comp_id", "label_name"], inplace=True)

    # 取没有概念标记的作为非概念标签的全集
    comp_nctag_table = pd.concat([data_raw_nctag_p1, data_raw_nctag_p2])
    comp_nctag_table = comp_nctag_table[~comp_nctag_table.label_name.isin(ctag_full_list)].reset_index(drop=True)
    comp_nctag_table["type"] = 1
    comp_tag = pd.concat([comp_ctag_table, comp_nctag_table]).reset_index(drop=True)
    # 将主营得分聚合
    comp_tag = comp_tag.groupby(["comp_id", "label_name"]).agg({"main_business_type": lambda x: sum(x) + 1, "type": max}).reset_index()
    pickle.dump(comp_tag, open("../Data/Output/Tag_graph/comp_tag_raw.pkl", "wb"))
    pickle.dump(label_chains_raw, open("../Data/Output/Tag_graph/label_chains_raw.pkl", "wb"))
    for x in locals().keys():
        del(locals()[x])
        gc.collect()
    return 0


def data_aggregator(nctag_filter_num=(150, 1000000)):
    comp_tag = pickle.load(open("../Data/Output/Tag_graph/comp_tag_raw.pkl", "rb"))
    # 为每一个公司赋予一个整数ID，以减小之后的计算量
    comp_id_dict = comp_tag["comp_id"].drop_duplicates().reset_index(drop=True)
    comp_id_dict = comp_id_dict.reset_index()
    comp_id_dict.rename(index=str, columns={"index": "comp_int_id"}, inplace=True)
    comp_tag = comp_tag.merge(comp_id_dict, how="left", left_on="comp_id", right_on="comp_id")


    # 将标签数据各自按照标签id进行聚合
    comp_tag["count_comps"] = 1
    tag_comps_aggregated = comp_tag.groupby("label_name") \
        .agg({"comp_int_id": lambda x: set(x), "count_comps": "count", "type": max}).reset_index()
    tag_comps_aggregated = tag_comps_aggregated[((tag_comps_aggregated.count_comps >= nctag_filter_num[0]) \
                                                 & (tag_comps_aggregated.count_comps <= nctag_filter_num[1])) \
                                                 | (tag_comps_aggregated.type == 0)]\
                                                 [["label_name", "comp_int_id", "type"]].reset_index(drop=True)

    # 为每一个标签赋予一个UUID，这个方式下，只要NAMESPACE不变，重复生成的也会是同一个UUID，避免了增量更新的麻烦
    tag_list = tag_comps_aggregated["label_name"].drop_duplicates().reset_index(drop=True)
    tag_list = tag_list.reset_index()
    tag_list.rename(index=str, columns={"index": "tag_uuid"}, inplace=True)
    tag_list.tag_uuid = tag_list.label_name.apply(lambda x: uuid.uuid5(uuid.NAMESPACE_URL, x).hex)
    comp_tag = comp_tag.merge(tag_list, how="left", left_on="label_name", right_on="label_name")
    tag_comps_aggregated = tag_comps_aggregated.merge(tag_list, how="left", left_on="label_name", right_on="label_name")
    comp_tag.dropna(how="any", inplace=True)
    tag_comps_aggregated.drop(["label_name"], axis=1, inplace=True)
    tag_comps_aggregated_file = "../Data/Output/Tag_graph/tag_comps_aggregated.pkl"
    pickle.dump(tag_comps_aggregated, open(tag_comps_aggregated_file, "wb"))
    tag_comps_aggregated["comp_list"] = tag_comps_aggregated.comp_int_id.apply(lambda x: "#".join(list(map(lambda y: str(y), x))))
    tag_comps_aggregated[["tag_uuid", "comp_list", "type"]].to_csv("../Data/Output/Tag_graph/tag_comps.csv", header=None, index=False)
    print("tag_comps_aggregated saved!")
    
    '''
    ###############################
    * 标签id-名称储存用作导入图数据库 *
    ###############################
    '''
    tag_points = comp_tag[["tag_uuid", "label_name", "type"]].drop_duplicates(subset=["tag_uuid"])
    tag_points.type = tag_points.type.apply(lambda x: "ctag" if x == 0 else "nctag")
    tag_points["point_type"] = "tag"
    tag_points.columns = header_dict.get("point")
    tag_points.to_csv("../Data/Output/Tag_graph/tag_points.points", header=None, index=False)
    # tag_points.to_sql(name=points_table, con=engine, if_exists="append", index=False)
    print("tag_points appended!")

    '''
    ###############################
    * 公司与标签关系储存用作入图数据库 *
    ###############################
    '''
    comp_tag_relations = comp_tag[["comp_id", "tag_uuid", "main_business_type"]] \
        .rename(columns={"main_business_type": "rel_value"}) \
        .drop_duplicates()
    # comp_tag_relations.rel_value = comp_tag_relations.rel_value.apply(lambda x: x + 1.0)
    # comp_tag_relations["rel_value"] = 1.0
    comp_tag_relations["rel_type"] = "company_tag"
    comp_tag_relations.columns = header_dict.get("relation")
    print("company-tag link count: %d" % len(comp_tag_relations))
    comp_tag_relations.to_csv("../Data/Output/Tag_graph/comp_tag_relations.relations", header=None, index=False)
    # comp_tag_relations.to_sql(name=relations_table, con=engine, if_exists="append", index=False)
    print("comp_tag_relations saved")

    
    # 将标签对应的hashcode以字典形式存成二进制文件
    tag_dict = dict(zip(tag_list.label_name, tag_list.tag_uuid))
    tag_dict_file = open("../Data/Output/Tag_graph/tag_dict.pkl", "wb")
    pickle.dump(tag_dict, tag_dict_file)
    tag_dict_file.close()
    
    # 将公司-标签UUID以二进制文件储存
    pickle.dump(comp_tag, open("../Data/Output/Tag_graph/comp_tag.pkl", "wb"))
    
    for x in locals().keys():
        del(locals()[x])
        gc.collect()
    return 0

def properties():
    comp_tag = pickle.load(open("../Data/Output/Tag_graph/comp_tag.pkl", "rb"))
    label_chains_raw = pickle.load(open("../Data/Output/Tag_graph/label_chains_raw.pkl", "rb"))
    comp_tags_aggregated = comp_tag.groupby("comp_id") \
        .apply(lambda x: set(zip(x.tag_uuid, x.main_business_type))) \
        .reset_index() \
        .rename(columns={0: "tag_list"})
    # comp_tags_aggregated.rename(columns={"tag_uuid": "comp_tag_list"}, inplace=True)
    comp_tags_aggregated.dropna(how="any", inplace=True)

    comp_tags_file_name = "../Data/Output/Tag_graph/comp_tags_all.pkl"
    comp_tags_all_file = open(comp_tags_file_name, "wb")
    pickle.dump(comp_tags_aggregated, comp_tags_all_file)
    comp_tags_all_file.close()
    
    # 储存概念标签的位置关系之后作为筛选属性
    label_chains_raw.rename(index=str, columns={"label_note_name":"label_node_name", "label_type_note":"label_type_node"}, inplace=True)
    tag_code_dict = pd.DataFrame.from_dict(pickle.load(open("../Data/Output/Tag_graph/tag_dict.pkl", "rb")), orient="index").reset_index()
    tag_code_dict.columns = ["label_name", "tag_code"]
    tag_code_root = tag_code_dict.rename(index=str, columns={"tag_code":"root_code", "label_name":"root_name"}, inplace=False)
    tag_code_node = tag_code_dict.rename(index=str, columns={"tag_code":"node_code", "label_name":"node_name"}, inplace=False)
    label_chains_link = label_chains_raw.merge(tag_code_node, how='left', left_on='label_node_name', right_on='node_name') \
        .merge(tag_code_root, how='left', left_on='label_root_name', right_on='root_name')
    label_chains_link["distance"] = label_chains_link.label_type_node - label_chains_link.label_type_root
    label_chains_link = label_chains_link[["node_code", "root_code", "distance"]]
    label_chains_link_reverse = label_chains_link[["root_code", "node_code", "distance"]]
    label_chains_link_reverse.columns = ["node_code", "root_code", "distance"]
    label_chains_link_reverse.distance = - label_chains_link_reverse.distance
    label_chains_all = pd.concat([label_chains_link, label_chains_link_reverse])
    label_self = label_chains_all.node_code.drop_duplicates().reset_index().rename(index=str, columns={"index": "distance"}, inplace=False)
    label_self.distance = 0
    label_self["root_code"] = label_self["node_code"]
    label_chains_all = pd.concat([label_chains_all, label_self]).dropna(how="any")
    label_chains_all["label_link"] = label_chains_all.node_code + "-" + label_chains_all.root_code
    label_chains_all[["label_link", "distance"]].to_csv("../Data/Output/Tag_graph/label_chains.csv", header=None, index=False)
    print("label_chains saved")
    
    for x in locals().keys():
        del(locals()[x])
        gc.collect()
    return 0

## 标签关系计算
def final_count(l1, l2):
    len1 = len(l1)
    len2 = len(l2)
    intersect = len(l1.intersection(l2))
    if intersect == 0:
        return 0
    else:
        return intersect/(len1 + len2 - intersect)

def simple_minmax(column_target, min_v=0.001, max_v=1):
    target = column_target.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(min_v, max_v))
    scaler.fit(target)
    return scaler.transform(target)


def cal_block_pair(block1, block2):
    tmp = block1.merge(block2, on="key")
    tmp["link_value"] = tmp[["comp_int_id_x", "comp_int_id_y"]].apply(lambda x: final_count(x[0], x[1]), axis=1)
    tmp["link_type"] = tmp[["type_x", "type_y"]].apply(lambda x: x[0] + x[1], axis=1)
    result_part = tmp[tmp.link_value != 0][["tag_uuid_x", "tag_uuid_y", "link_type", "link_value"]]
    # result_part.to_csv("../Data/Output/Tag_graph/temp_result/part_result_%d_%d.relations" % (k + 1, i + 1), index=False, header=None)
    return result_part

def multiprocess_block(block1, whole_data, process_list, k, process_num=8):
    result_list = []
    pool = mp.Pool()
    for p in range(0, process_num):
        result_list.append(pool.apply_async(cal_block_pair, (block1, whole_data.iloc[process_list[p].tolist()])))
    pool.close()
    pool.join()
    result_merged = pd.concat([r.get() for r in result_list])
    result_merged.to_csv("../Data/Output/Tag_graph/temp_result/part_result_%d.relations" % (k + 1), index=False, header=None)
    
    for x in locals().keys():
        del(locals()[x])
        gc.collect()

# 标签两两关系计算
def tag_tag4(batch_size=5, process_num=8):
    tag_comps_aggregated = pickle.load(open("../Data/Output/Tag_graph/tag_comps_aggregated.pkl", "rb"))
    print("Data loaded")
    print("------ Start tag tag calculation ------")
    tag_comps_aggregated["key"] = 1
    row_num = len(tag_comps_aggregated)
    index_list = np.array_split(tag_comps_aggregated.index, batch_size)
    process_list = np.array_split(tag_comps_aggregated.index, process_num)
    for k in tqdm(range(0, batch_size)):
        epoch_start_time = datetime.datetime.now()
        print("*** Start epoch %d at %s ***" % (k + 1, epoch_start_time.strftime('%H:%M:%S')))
        multiprocess_block(tag_comps_aggregated.iloc[index_list[k].tolist()] , tag_comps_aggregated, process_list, k, process_num=process_num)
        epoch_end_time = datetime.datetime.now()
        print("*** Epoch %d finished at %s (time used: %.3f seconds) ***" % (k + 1, epoch_end_time.strftime('%H:%M:%S'), (epoch_end_time - epoch_start_time).total_seconds()))
        for x in locals().keys():
            del(locals()[x])
            gc.collect()
    print("------ All tag_tag calculation done ------")
    os.system("cat ../Data/Output/Tag_graph/temp_result/part_result_* > ../Data/Output/Tag_graph/temp_result/tag_tag_result_all")
    tag_tag = pd.read_csv("../Data/Output/Tag_graph/temp_result/tag_tag_result_all", header=None)
    tag_tag.columns = ["tag1", "tag2", "link_type", "link_value"]
    tag_tag.link_value = tag_tag.link_value.apply(lambda x: np.log2(min(0.000001 + x, 1)))
    tag_tag.link_value = simple_minmax(tag_tag.link_value)
    tag_tag = tag_tag[["tag1", "tag2", "link_type", "link_value"]].drop_duplicates()
    tag_tag["tag_link"] = tag_tag.tag1 + "-" + tag_tag.tag2
    tag_tag_dict = dict(zip(tag_tag.tag_link, (tag_tag.link_value, tag_tag.link_type)))

    pickle.dump(tag_tag_dict, open("../Data/Output/Tag_graph/tag_tag.pkl", "wb"))

    
    '''
    ################################
    * 概念-概念标签关系用于导入图数据库 *
    ################################
    '''
    # 过滤同链标签
    label_chains_all = pd.read_csv("../Data/Output/Tag_graph/label_chains.csv", header=None)
    label_chains_all.columns = ["node_link", "distance"]
    label_chains_drop = label_chains_all[label_chains_all.distance != 1]
    node_chains = label_chains_all[label_chains_all.distance == 1]
    tag_tag_relations = tag_tag[~tag_tag.tag_link.isin(label_chains_drop.node_link)][["tag1", "tag2", "link_value", "link_type", "tag_link"]].copy()

    tag_tag_relations.rename(index=str, columns={"link_type": "rel_type"}, inplace=True)
    link_type_dict = {0: "ctag_ctag", 1: "ctag_nctag", 2: "nctag_nctag"}
    
    tag_tag_relations.rel_type = tag_tag_relations.rel_type.apply(lambda x: link_type_dict.get(x))
    tag_tag_relations.loc[tag_tag_relations.tag_link.isin(node_chains.node_link), "rel_type"] = "node_of"
    tag_tag_relations.drop(["tag_link"], axis=1, inplace=True)
    tag_tag_relations.columns = header_dict.get("relation")
    
    tag_tag_relations.to_csv("../Data/Output/Tag_graph/tag_tag_relations.relations")
    
    print("tag-tag link records count: %d" % len(tag_tag_relations))
    for x in locals().keys():
        del(locals()[x])
        gc.collect()
    return 0Z
        
def dense_to_sparse(tag_tag_weights=(0.5, 0.3, 0.2)):
    tag_tag = pd.read_csv("../Data/Output/Tag_graph/temp_result/tag_tag_result_all", header=None)
    tag_dict = tag_dict = pickle.load(open("../Data/Output/Tag_graph/tag_dict.pkl", "rb"))
    tag_tag.columns = ["tag1", "tag2", "link_type", "link_value"]
    tag_tag.link_value = tag_tag.link_value.apply(lambda x: np.log2(min(0.000001 + x, 1)))
    tag_tag.link_value = simple_minmax(tag_tag.link_value)
    tag_tag = tag_tag[["tag1", "tag2", "link_type", "link_value"]].drop_duplicates()
    tag_index = dict(zip(tag_dict.values(), range(0, len(tag_dict))))
    length = len(tag_index)
    row = np.array(tag_tag.tag1.apply(lambda x: tag_index.get(x)))
    col = np.array(tag_tag.tag2.apply(lambda x: tag_index.get(x)))
    data = np.array(tag_tag.link_value * tag_tag.link_type.apply(lambda x: tag_tag_weights[x]))
    tag_tag_sparse = coo_matrix((data, (row, col)), shape=(length, length))
    pickle.dump(tag_tag_sparse, open("../Data/Output/Tag_graph/tag_tag_sparse.pkl", "wb"))
    
    comp_tags_all = pickle.load(open("../Data/Output/Tag_graph/comp_tags_all.pkl", "rb"))
    comp_tags_all["row_num"] = comp_tags_all.index
    comp_tags_all["vector_len"] = comp_tags_all.comp_tag_list.apply(lambda x: len(x))
    comp_tags_all["coo"]= comp_tags_all[["comp_tag_list", "row_num", "vector_len"]].apply(lambda x: [(x[1], tag_index.get(w[0]), w[1]/x[3]) for w in x[0]], axis=1)
    coord_list = [x for y in comp_tags_all.coo for x in y]
    coords = tuple(map(list, zip(*coord_list)))
    comp_tags_all_sparse = coo_matrix((coords[2], (coords[0], coords[1])), shape=(len(comp_tags_all), length))
    pickle.dump(comp_tags_all_sparse, open("../Data/Output/Tag_graph/comp_tags_all_sparse", "wb"))
    
    
# 需要导入neo4j的数据进行合并
def neo4j_merge():
    # 全部点集合
    tag_points = pd.read_csv("../Data/Output/Tag_graph/tag_points.points", header=None)
    company_points = pd.read_csv("../Data/Output/Tag_graph/company_points.points", header=None)
    points = pd.concat([tag_points, company_points]).drop_duplicates().reset_index(drop=True)
    points.columns = header_dict["point"]
    points.reset_index(drop=True, inplace=True)
    points.rename(index=str, columns={"index": "id"}, inplace=True)
    points.fillna("", inplace=True)
    points.drop_duplicates(subset=["point_id"], inplace=True)
    points.reset_index(drop=True, inplace=True)
    # points.id = points.id.apply(lambda x: x + 1)
    points.to_csv("../Data/Output/Tag_graph/all_points.csv", index=False, header=None)
    print("Points saved!")

    # 边数据整合
    comp_tag_relations = pd.read_csv("../Data/Output/Tag_graph/comp_tag_relations.relations", header=None)
    tag_tag_relations = pd.read_csv("../Data/Output/Tag_graph/tag_tag_relations.relations", header=None)
    comp_tag_relations.columns = header_dict["relation"]
    tag_tag_relations.columns = header_dict["relation"]
    
    tag_tag_relations["link"] = tag_tag_relations[["src_id", "target_id"]].apply(lambda x: "-".join(sorted([str(x[0]), str(x[1])])), axis=1)
    tag_tag_relations.drop_duplicates(subset=["link"], inplace=True)
    tag_tag_relations.drop(["link"], axis=1, inplace=True)
    tag_tag_relations.drop(index=tag_tag_relations[tag_tag_relations.src_id == tag_tag_relations.target_id].index, inplace=True)
    
    all_relations = pd.concat([tag_tag_relations, comp_tag_relations]) \
        .drop_duplicates().reset_index(drop=True) 

    all_relations.to_csv("../Data/Output/Tag_graph/all_relations.csv", index=False, header=None)
    print("Relations saved!")
    
    for x in locals().keys():
        del(locals()[x])
        gc.collect()
    return 0


def to_graph_database():
    stop_neo4j = os.system("neo4j stop")
    rm_old_version = os.system("rm -rf ../Data/Output/Tag_graph/to_neo4j/graph.db.last_version")
    backup_old_version = os.system("mv ../Data/Output/Tag_graph/to_neo4j/graph.db ../Data/Output/Tag_graph/to_neo4j/graph.db.last_version")
    import_neo4j = os.system("neo4j-import  --into ../Data/Output/Tag_graph/to_neo4j/graph.db --multiline-fields=true --bad-tolerance=1000 --id-type string --nodes:points ../Data/Input/Tag_graph/points_header.csv,../Data/Output/Tag_graph/all_points.csv --relationships:relations ../Data/Input/Tag_graph/relations_header.csv,../Data/Output/Tag_graph/all_relations.csv")
    if import_neo4j == 0:
        print("Data imported to neo4j!")
        start_neo4j = os.system("nohup neo4j start > /data1/zijun.wu/neo4j.log 2>&1 &")
    else:
        print("Import to neo4j failed!")
    return 0
