# -*- coding: utf-8 -*-
#%%
import pandas as pd
import numpy as numpy
import uuid
import pickle
import os
import configparser
import pymysql
import datetime
import numpy as np
import gc
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

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

header_dict = {
    "point": ["point_id", "name", "property", "point_type"],
    "relation": ["src_id", "target_id", "rel_value", "rel_type"]
}


# 基本数据处理及生成
def comp_tag(new_result="company_tag_info_latest0622", old_result="company_tag", label_code_relation="label_code_relation", keep_list=False, db=db):
    # 从库中读取数据
    sql_new_result = "select comp_id, comp_full_name, label_name, classify_id, label_type, label_type_num from %s" % new_result
    sql_old_result = "select comp_id, comp_full_name, key_word from %s" % old_result
    sql_label_code_relation = "select * from %s" % label_code_relation
    data_raw_new = pd.read_sql(sql_new_result, con=db)
    data_raw_old_full = pd.read_sql(sql_old_result, con=db)
    label_chains_raw = pd.read_sql(sql_label_code_relation, con=db)
    data_raw_new.fillna("", inplace=True)
    data_raw_old_full.fillna("", inplace=True)
    # 生成公司id-name字典保存
    comp_id_name = pd.concat([data_raw_new[["comp_id", "comp_full_name"]], data_raw_old_full[["comp_id", "comp_full_name"]]]).drop_duplicates()

    comp_id_name_dict = dict(zip(comp_id_name.comp_id, comp_id_name.comp_full_name))
    comp_id_name_dict_file_name = "../Data/Output/Tag_graph/comp_id_name_dict.pkl"
    comp_id_name_dict_file = open(comp_id_name_dict_file_name, "wb")
    pickle.dump(comp_id_name_dict, comp_id_name_dict_file)
    comp_id_name_dict_file.close()
    print("comp_id_name_dict saved!")
    
    '''
    ###############################
    * 公司id-名称储存用作导入图数据库 *
    ###############################
    '''
    company_points = comp_id_name[["comp_id", "comp_full_name"]]
    # company_points.comp_full_name = company_points.comp_full_name.apply(lambda x: x.strip().replace("(","（").replace(")","）"))
    company_points.drop_duplicates(inplace=True)
    company_points["property"] = ""
    company_points["point_type"] = "company"
    company_points.columns = header_dict.get("point")
    
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
    comp_ctag_table = comp_ctag_table_all_infos[["comp_id", "label_name"]].reset_index(drop=True)
    comp_ctag_table["type"] = 0
    
    # 新系统下结果中的公司-非概念标签
    data_raw_nctag_p1 = data_raw_new[data_raw_new.classify_id == 4][["comp_id", "label_name"]]

    # 读取旧版数据，只取其中的非概念标签（概念标签无法确定其层级和产业链（复用））
    data_raw_nctag_p2_raw = data_raw_old_full[data_raw_old_full.key_word != ""][["comp_id", "key_word"]]
    data_raw_nctag_p2_raw.dropna(subset=["comp_id", "key_word"], inplace=True)
    data_raw_nctag_p2_raw.columns = ["comp_id", "label_name"]

    # 新版的非概念标签和旧版整体数据拼接后进行split和flatten
    tuples = data_raw_nctag_p2_raw.apply(lambda x: [(x[0], t) for t in x[1].split(",") if t != ""], axis=1)
    flatted = [y for x in tuples for y in x]
    data_raw_nctag_p2 = pd.DataFrame(flatted, columns=["comp_id", "label_name"]).drop_duplicates()

    # 取没有概念标记的作为非概念标签的全集
    comp_nctag_table = pd.concat([data_raw_nctag_p1, data_raw_nctag_p2])
    comp_nctag_table = comp_nctag_table[~comp_nctag_table.label_name.isin(ctag_full_list)].reset_index(drop=True)
    comp_nctag_table["type"] = 1
    comp_tag = pd.concat([comp_ctag_table, comp_nctag_table]).reset_index(drop=True)
    del(comp_ctag_table_all_infos)
    del(comp_ctag_table)
    del(comp_nctag_table)
    gc.collect()
    return (comp_tag, company_points, label_chains_raw)


def data_aggregator(comp_tag, label_chains_raw, nctag_filter_num=(150, 1000000)):
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
                                                 [["label_name", "comp_int_id"]].reset_index(drop=True)

    # 为每一个标签赋予一个UUID，这个方式下，只要NAMESPACE不变，重复生成的也会是同一个UUID，避免了增量更新的麻烦
    tag_list = tag_comps_aggregated["label_name"].drop_duplicates().reset_index(drop=True)
    tag_list = tag_list.reset_index()
    tag_list.rename(index=str, columns={"index": "tag_uuid"}, inplace=True)
    tag_list.tag_uuid = tag_list.label_name.apply(lambda x: uuid.uuid5(uuid.NAMESPACE_URL, x).hex)
    comp_tag = comp_tag.merge(tag_list, how="left", left_on="label_name", right_on="label_name")
    tag_comps_aggregated = tag_comps_aggregated.merge(tag_list, how="left", left_on="label_name", right_on="label_name")
    
    '''
    ###############################
    * 标签id-名称储存用作导入图数据库 *
    ###############################
    '''
    tag_points = comp_tag[["tag_uuid", "label_name", "type"]]
    tag_points.type = tag_points.type.apply(lambda x: "ctag" if x == 0 else "nctag")
    tag_points["point_type"] = "tag"
    tag_points.columns = header_dict.get("point")

    '''
    ###############################
    * 公司与标签关系储存用作入图数据库 *
    ###############################
    '''
    comp_tag_relations = comp_tag[["comp_id", "tag_uuid"]]
    comp_tag_relations["rel_value"] = 1.0
    comp_tag_relations["rel_type"] = "company_tag"
    comp_tag_relations.columns = header_dict.get("relation")
    print("company-tag link count: %d" % len(comp_tag_relations))
    
    # comp_tag.drop(["label_name", "comp_id"], axis=1, inplace=True)

    
    # 将标签对应的hashcode以字典形式存成二进制文件
    tag_dict = dict(zip(tag_list.label_name, tag_list.tag_uuid))
    tag_dict_file = open("../Data/Output/Tag_graph/tag_dict.pkl", "wb")
    pickle.dump(tag_dict, tag_dict_file)
    tag_dict_file.close()
    
    comp_tags_aggregated = comp_tag.pivot_table(index="comp_id", columns=["type"], aggfunc={"tag_uuid": lambda x: set(x)})
    comp_tags_aggregated.columns = comp_tags_aggregated.columns.levels[1]
    comp_tags_aggregated.rename(index=int, columns={0: "ctag", 1: "nctag"}, inplace=True)
    comp_tags_aggregated.reset_index(inplace=True)
    
    comp_tags_aggregated.fillna(0, inplace=True)
    comp_tags_aggregated.ctag = comp_tags_aggregated.ctag.apply(lambda x: set([]) if x == 0 else x)
    comp_tags_aggregated.nctag = comp_tags_aggregated.nctag.apply(lambda x: set([]) if x == 0 else x)
    comp_tags_aggregated["tag_infos"] = comp_tags_aggregated[["ctag", "nctag"]].apply(lambda x: {"ctag": x[0], "nctag": x[1]}, axis=1)
    # comp_tags_aggregated = comp_tags_aggregated.merge(comp_id_dict, how="left", left_on="comp_int_id", right_on="comp_int_id")
    comp_tags_aggregated.drop(["ctag", "nctag"], axis=1, inplace=True)
    comp_tags_all_dict = dict(zip(comp_tags_aggregated.comp_id, comp_tags_aggregated.tag_infos))
    comp_tags_file_name = "../Data/Output/Tag_graph/comp_tags_all.pkl"
    comp_tags_all_file = open(comp_tags_file_name, "wb")
    pickle.dump(comp_tags_all_dict, comp_tags_aggregated)
    comp_tags_all_file.close()

    # 储存概念标签的位置关系之后作为筛选属性
    ctag_position_file_name = "../Data/Output/Tag_graph/ctag_position.pkl"
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
    ctag_position_dict = dict(zip(label_chains_all.label_link, label_chains_all.distance))
    ctag_position_file = open(ctag_position_file_name, "wb")
    pickle.dump(ctag_position_dict, ctag_position_file)
    ctag_position_file.close()
    del(comp_tag)
    return (tag_comps_aggregated, tag_points, comp_tag_relations)

## 标签关系计算
def final_count(l1, l2):
    if len(l1.union(l2)) == 0:
        return 0
    else:
        return len(l1.intersection(l2))/len(l1.union(l2))

def simple_minmax(column_target, min_v=0.001, max_v=1):
    target = column_target.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(min_v, max_v))
    scaler.fit(target)
    return scaler.transform(target)


# 标签两两关系计算
def tag_tag(tag_comps_aggregated):
    tag_comps_aggregated["key"] = 1
    tag_tag = nctag_comps_aggregated.merge(nctag_comps_aggregated, on="key")
    record_len = len(tag_tag)
    interval_size = record_len // 20
    i = 0
    # 由于非概念标签之间的计算量级较大，服务器内存不足，因此采用分块计算、最后融合的方式
    while interval_size*i < record_len:
        start_time = datetime.datetime.now()
        print("### start tag-tag part %d at %s ###" % (i, start_time.strftime('%H:%M:%S')))
        tmp = tag_tag[interval_size*i:min(interval_size * (i + 1), record_len)].copy()
        tmp["link_value"] = tmp[["comp_int_id_x", "comp_int_id_y"]].apply(lambda x: x[2] * x[3] * final_count(x[0], x[1]), axis=1)
        tmp["link_type"] = tmp[["type_x", "type_y"]].apply(lambda x: x[0] + x[1])
        result_part = tmp[tmp.link_value != 0][["tag_uuid_x", "tag_uuid_y", "link_type", "link_value"]]
        result_part.to_csv("../Data/Output/Tag_graph/temp_result/part_result_%d.relations" % i, index=False, header=None)
        end_time = datetime.datetime.now()
        print("### Part %d finished at %s (time used: %.3f seconds) ###" % (i, end_time.strftime('%H:%M:%S'), (end_time - start_time).total_seconds()))
        i += 1
    print("tag-tag calculation done")
    os.system("cat ../Data/Output/Tag_graph/temp_result/part_result_* > ../Data/Output/Tag_graph/temp_result/tag_tag_result_all")
    tag_tag = pd.read_csv("../Data/Output/Tag_graph/temp_result/tag_tag_result_all", header=None)
    tag_tag.columns = ["tag1", "tag2", "link_type", "link_value"]
    tag_tag.link_value = nctag_nctag.link_value.apply(lambda x: np.log2(min(0.000001 + x, 1)))
    tag_tag.link_value = simple_minmax(nctag_nctag.link_value)
    tag_tag = tag_tag[["tag1", "tag2", "link_type", "link_value"]].drop_duplicates()
    tag_tag["tag_link"] = tag_tag.tag1 + "-" + tag_tag.tag2
    tag_tag_dict = dict(zip(tag_tag.tag_link, (tag_tag.link_value, tag_tag.link_type)))
    tag_tagg_file_name = "../Data/Output/Tag_graph/tag_tag.pkl"
    tag_tag_file = open(tag_tag_file_name, "wb")
    pickle.dump(tag_tag_dict, tag_tag_file)
    tag_tag_file.close()
    
    
    '''
    ################################
    * 概念-概念标签关系用于导入图数据库 *
    ################################
    '''
    # 过滤同链标签
    ctag_position = pickle.load(open("../Data/Output/Tag_graph/ctag_position.pkl", "rb"))
    label_chains_all = pd.DataFrame.from_dict(ctag_position, orient="index").reset_index()
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
    print("tag-tag link records count: %d" % len(tag_tag_relations))
    return tag_tag_relations


# 需要导入neo4j的数据进行合并
def neo4j_merge(tag_points, company_points, comp_tag_relations, tag_tag_relations):
    # 全部点集合
    points = pd.concat([tag_points, company_points]).drop_duplicates().reset_index(drop=True)
    points.columns = header_dict["point"]
    points.reset_index(drop=True, inplace=True)
    points.rename(index=str, columns={"index": "id"}, inplace=True)
    points.fillna("", inplace=True)
    points.drop_duplicates(subset=["point_id"], inplace=True)
    points = points.groupby("point_id").agg("min").reset_index()
    # points.id = points.id.apply(lambda x: x + 1)
    points.to_csv("../Data/Output/Tag_graph/all_points.csv", index=False, header=None)
    print("Points saved!")

    # 边数据整合
    all_relations = pd.concat([tag_tag_relations, comp_tag_relations]) \
        .drop_duplicates().reset_index(drop=True)
    all_relations.columns = header_dict["relation"]
    all_relations["link"] = all_relations[["src_id", "target_id"]].apply(lambda x: "-".join(sorted([str(x[0]), str(x[1])])), axis=1)
    all_relations.drop_duplicates(subset=["link"], inplace=True)
    all_relations.drop(["link"], axis=1, inplace=True)
    all_relations = all_relations[["src_id", "target_id", "rel_value", "rel_type"]]
    all_relations.drop(index=all_relations[all_relations.src_id == all_relations.target_id].index)
    all_relations.reset_index(drop=True, inplace=True)
    all_relations.to_csv("../Data/Output/Tag_graph/all_relations.csv", index=False, header=None)
    print("Relations saved!")
    return 0

def to_graph_database(method="replace"):
    import_neo4j = os.system("neo4j-import  --into graph.db --multiline-fields=true --bad-tolerance=1000 --id-type string --nodes:points ../Data/Output/Tag_graph/points_header.csv,../Data/Output/Tag_graph/to_neo4j/all_points.csv --relationships:relations ../Data/Output/Tag_graph/relations_header.csv,../Data/Output/Tag_graph/to_neo4j/all_relations.csv")
    if import_neo4j == 0:
        print("Data imported to neo4j!")
        os.system("mv -rf graph.db /home/zijun.wu/my_lib/neo4j-community-3.4.0/data/databases/")
    else:
        print("Import to neo4j failed!")
    return 0