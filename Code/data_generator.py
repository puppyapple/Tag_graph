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
def comp_tag(new_result="company_tag_info_latest", old_result="company_tag", label_code_relation="label_code_relation", keep_list=False, db=db):
    # 从库中读取数据
    sql_new_result = "select * from %s" % new_result
    sql_old_result = "select * from %s" % old_result
    sql_label_code_relation = "select * from %s" % label_code_relation
    data_raw_new = pd.read_sql(sql_new_result, con=db)
    data_raw_old_full = pd.read_sql(sql_old_result, con=db)
    label_chains_raw = pd.read_sql(sql_label_code_relation, con=db)
    data_raw_new.fillna("", inplace=True)
    data_raw_old_full.fillna("", inplace=True)
    # 生成公司id-name字典保存
    comp_id_name = pd.concat([data_raw_new[["comp_id", "comp_full_name"]], data_raw_old_full[["comp_id", "comp_full_name"]]]).drop_duplicates()
    # print(comp_id_name.head(10))
    # comp_id_name = comp_id_name[comp_id_name.comp_id.isin(comps_to_concern)]
    comp_id_name_dict = dict(zip(comp_id_name.comp_id, comp_id_name.comp_full_name))
    comp_id_name_dict_file_name = "../Data/Output/Tag_graph/comp_id_name_dict.pkl"
    comp_id_name_dict_file = open(comp_id_name_dict_file_name, "wb")
    pickle.dump(comp_id_name_dict, comp_id_name_dict_file)
    comp_id_name_dict_file.close()
    
    '''
    ###############################
    * 公司id-名称储存用作导入图数据库 *
    ###############################
    '''
    company_points = comp_id_name[["comp_id", "comp_full_name"]].copy()
    company_points.comp_full_name = company_points.comp_full_name.apply(lambda x: x.strip().replace("(","（").replace(")","）"))
    company_points.drop_duplicates(inplace=True)
    company_points["property"] = ""
    company_points["point_type"] = "company"
    company_points.columns = header_dict.get("point")
    
    # 全部概念标签的列表
    ctag_full_list = set(label_chains_raw.label_note_name).union(set(label_chains_raw.label_root_name))
    # chains_to_keep = label_chains_raw[label_chains_raw.label_root_name.isin(keep_list)]
    # tags_to_keep = set(chains_to_keep.label_note_name).union(set(chains_to_keep.label_root_name))
    
    if keep_list == True:
        keep_list = ctag_keep_list
    else:
        keep_list = ctag_full_list
    
    # 根据输入的公司概念和非概念标记源数据，分别得到完整的公司-概念标签、公司-非概念标签
    data_raw_new.dropna(subset=["comp_id", "label_name"], inplace=True)
    cols = ["comp_id", "comp_full_name", "label_name", "classify_id", "label_type", "label_type_num", "src_tags", "remarks"]
    data_raw_new = data_raw_new[data_raw_new.label_name != ''][cols].copy()
    
    # 将技术标签对应的标签作处理，在标签前加上前缀以示区别（和非概念标签），之后由分析师方面统一解决
    data_raw_new.label_name = data_raw_new[["label_name", "remarks"]].apply(lambda x: "技术标签_" + x[0] if x[1] == "1" else x[0], axis=1)
    
    # 过滤掉进行中的产业链话题等关联的公司
    comp_ctag_table_all_infos_raw = data_raw_new[data_raw_new.classify_id != 4].reset_index(drop=True)
    comp_ctag_table_all_infos = comp_ctag_table_all_infos_raw.copy()
    comp_ctag_table_all_infos.src_tags = comp_ctag_table_all_infos[["label_type_num", "src_tags"]].apply(lambda x: x[1].split("#")[x[0] - 1], axis=1)
    comp_ctag_table_all_infos = comp_ctag_table_all_infos[comp_ctag_table_all_infos.src_tags.apply(lambda x: x.split("-")[0]).isin(keep_list)]
    comp_ctag_table = comp_ctag_table_all_infos[["comp_id", "label_name"]].reset_index(drop=True)
    
    # 概念标签全集列表，加上一列1作为标记用
    # ctag_list = comp_ctag_table_all_infos.label_name.drop_duplicates().reset_index()
    # ctag_list.rename(index=str, columns={"index": "ctag_mark"}, inplace=True)
    # ctag_list.ctag_mark = ctag_list.ctag_mark.apply(lambda x: 1)

    # 新系统下结果中的公司-非概念标签
    data_raw_nctag_p1 = data_raw_new[data_raw_new.classify_id == 4][["comp_id", "label_name"]].copy()

    # 读取旧版数据，只取其中的非概念标签（概念标签无法确定其层级和产业链（复用））
    data_raw_old = data_raw_old_full[["comp_id", "key_word"]].copy()
    data_raw_old.dropna(subset=["comp_id", "key_word"], inplace=True)
    data_raw_old.columns = ["comp_id", "label_name"]
    data_raw_old = data_raw_old[data_raw_old.label_name != ""].copy()

    # 新版的非概念标签和旧版整体数据拼接后进行split和flatten
    data_to_flatten = pd.concat([data_raw_old, data_raw_nctag_p1])
    tuples = data_to_flatten.apply(lambda x: [(x[0], t) for t in x[1].split(",") if t != ""], axis=1)
    flatted = [y for x in tuples for y in x]
    data_raw_nctag_flatted = pd.DataFrame(flatted, columns=["comp_id", "label_name"]).drop_duplicates()
    # data_raw_nctag_with_mark = data_raw_nctag_flatted.merge(ctag_list, how="left", left_on="label_name", right_on="label_name")

    # 取没有概念标记的作为非概念标签的全集
    comp_nctag_table = data_raw_nctag_flatted[~data_raw_nctag_flatted.label_name.isin(ctag_full_list)].reset_index(drop=True)

    return (comp_ctag_table, comp_nctag_table, company_points, label_chains_raw, comp_ctag_table_all_infos_raw)


def data_aggregator(comp_ctag_table, comp_nctag_table, company_points, label_chains_raw, nctag_filter_num=50):
    comp_tag_table_all = pd.concat([comp_ctag_table, comp_nctag_table])
    
    # 为每一个公司赋予一个整数ID，以减小之后的计算量
    comp_id_dict = comp_tag_table_all["comp_id"].drop_duplicates().reset_index(drop=True)
    comp_id_dict = comp_id_dict.reset_index()
    comp_id_dict.rename(index=str, columns={"index": "comp_int_id"}, inplace=True)
    comp_ctag_table = comp_ctag_table.merge(comp_id_dict, how="left", left_on="comp_id", right_on="comp_id")
        #.drop(["comp_id"], axis=1)
    comp_nctag_table = comp_nctag_table.merge(comp_id_dict, how="left", left_on="comp_id", right_on="comp_id")
        #.drop(["comp_id"], axis=1)
    comp_total_num = len(comp_id_dict)

    # 为每一个标签赋予一个UUID，这个方式下，只要NAMESPACE不变，重复生成的也会是同一个UUID，避免了增量更新的麻烦
    tag_list = comp_tag_table_all["label_name"].drop_duplicates().reset_index(drop=True)
    tag_list = tag_list.reset_index()
    tag_list.rename(index=str, columns={"index": "tag_uuid"}, inplace=True)
    tag_list.tag_uuid = tag_list.label_name.apply(lambda x: uuid.uuid5(uuid.NAMESPACE_URL, x).hex)
    comp_ctag_table = comp_ctag_table.merge(tag_list, how="left", left_on="label_name", right_on="label_name")
    comp_nctag_table = comp_nctag_table.merge(tag_list, how="left", left_on="label_name", right_on="label_name")
    
    '''
    ###############################
    * 标签id-名称储存用作导入图数据库 *
    ###############################
    '''
    tag_c = comp_ctag_table[["tag_uuid", "label_name"]].copy().drop_duplicates()
    tag_c["property"] = "ctag"
    
    tag_nc = comp_nctag_table[["tag_uuid", "label_name"]].copy().drop_duplicates()
    tag_nc["property"] = "nctag"
    
    tag_points = pd.concat([tag_c, tag_nc]).drop_duplicates()
    tag_points["point_type"] = "tag"
    tag_points.columns = header_dict.get("point")

    '''
    ###############################
    * 公司与标签关系储存用作入图数据库 *
    ###############################
    '''
    comp_tag_relations = pd.concat([comp_ctag_table[["comp_id", "tag_uuid"]], comp_nctag_table[["comp_id", "tag_uuid"]]])
    comp_tag_relations["rel_value"] = 1.0
    comp_tag_relations["rel_type"] = "company_tag"
    comp_tag_relations.columns = header_dict.get("relation")
    print("company-tag link count: %d" % len(comp_tag_relations))
    
    comp_ctag_table.drop(["label_name", "comp_id"], axis=1, inplace=True)
    comp_nctag_table.drop(["label_name", "comp_id"], axis=1, inplace=True)
    
    # 将标签对应的hashcode以字典形式存成二进制文件
    tag_dict = dict(zip(tag_list.label_name, tag_list.tag_uuid))
    tag_dict_file = open("../Data/Output/Tag_graph/tag_dict.pkl", "wb")
    pickle.dump(tag_dict, tag_dict_file)
    tag_dict_file.close()

    # 将概念非概念标签数据各自按照标签id进行聚合
    ctag_comps_aggregated = comp_ctag_table.groupby("tag_uuid").agg({"comp_int_id": lambda x: set(x)}).reset_index()
    comp_nctag_table["count_comps"] = 1
    nctag_comps_aggregated_all = comp_nctag_table.groupby("tag_uuid") \
        .agg({"comp_int_id": lambda x: set(x), "count_comps": "count"}).reset_index()
    nctag_comps_aggregated = nctag_comps_aggregated_all[nctag_comps_aggregated_all.count_comps >= nctag_filter_num] \
        [["tag_uuid", "comp_int_id"]].reset_index(drop=True)

    
    comp_tags_file_name = "../Data/Output/Tag_graph/comp_tags_all.pkl"
    comp_ctags_aggregated = comp_ctag_table.groupby("comp_int_id").agg({"tag_uuid": lambda x: set(x)}).reset_index()
    comp_nctags_aggregated = comp_nctag_table.groupby("comp_int_id").agg({"tag_uuid": lambda x: set(x)}).reset_index()
    comp_ctags_aggregated.tag_uuid = comp_ctags_aggregated.tag_uuid.apply(lambda x: {"ctags": x})
    comp_nctags_aggregated.tag_uuid = comp_nctags_aggregated.tag_uuid.apply(lambda x: {"nctags": x})
    comp_tags_all = comp_ctags_aggregated.merge(comp_nctags_aggregated, how="outer", left_on="comp_int_id", right_on="comp_int_id")
    comp_tags_all.fillna(0, inplace=True)
    comp_tags_all.tag_uuid_x = comp_tags_all.tag_uuid_x.apply(lambda x: {} if x == 0 else x)
    comp_tags_all.tag_uuid_y = comp_tags_all.tag_uuid_y.apply(lambda x: {} if x == 0 else x)
    comp_tags_all["tag_infos"] = comp_tags_all[["tag_uuid_x", "tag_uuid_y"]].apply(lambda x: {**(x[0]), **(x[1])}, axis=1)
    comp_tags_all = comp_tags_all.merge(comp_id_dict, how="left", left_on="comp_int_id", right_on="comp_int_id")
    comp_tags_all.drop(["tag_uuid_x", "tag_uuid_y", "comp_int_id"], axis=1, inplace=True)
    comp_tags_all_dict = dict(zip(comp_tags_all.comp_id, comp_tags_all.tag_infos))
    comp_tags_all_file = open(comp_tags_file_name, "wb")
    pickle.dump(comp_tags_all_dict, comp_tags_all_file)
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
    label_chains_link = label_chains_link[["node_code", "root_code", "distance"]].copy()
    label_chains_link_reverse = label_chains_link[["root_code", "node_code", "distance"]].copy()
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
    return (ctag_comps_aggregated, nctag_comps_aggregated, tag_points, company_points, comp_tag_relations, comp_total_num)

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

# 考虑分开计算三组关系，因为非概念之间的内存占用可能较大
# 先计算每个非概念标签的伪TF-IDF系数
def nctag_idf(nctag_comps_aggregated, comp_total_num):
    nctag_idf = nctag_comps_aggregated[["tag_uuid"]].copy()
    nctag_idf["idf"] = nctag_comps_aggregated.comp_int_id.apply(lambda x: np.log2(comp_total_num/len(x)))
    nctag_idf.idf = simple_minmax(nctag_idf.idf)
    return nctag_idf

# 概念标签两两关系计算
def ctag_relation(ctag_comps_aggregated):
    ctag_comps_aggregated["key"] = 1
    ctag_ctag = ctag_comps_aggregated.merge(ctag_comps_aggregated, on="key")
    ctag_ctag["link_value"] = ctag_ctag[["comp_int_id_x", "comp_int_id_y"]].apply(lambda x: final_count(x[0], x[1]), axis=1)
    ctag_ctag = ctag_ctag[ctag_ctag.link_value != 0].copy()
    
    ctag_ctag.link_value = ctag_ctag.link_value.apply(lambda x: np.log2(min(0.000001 + x, 1)))
    ctag_ctag.link_value = simple_minmax(ctag_ctag.link_value)
    ctag_ctag = ctag_ctag[["tag_uuid_x", "tag_uuid_y", "link_value"]].drop_duplicates().copy()
    ctag_ctag["tag_link"] = ctag_ctag.tag_uuid_x + "-" + ctag_ctag.tag_uuid_y
    
    # 概念标签关联储存作为推荐系统读取使用
    ctag_ctag_dict = dict(zip(ctag_ctag.tag_link, ctag_ctag.link_value))
    ctag_ctag_file_name = "../Data/Output/Tag_graph/ctag_ctag.pkl"
    ctag_ctag_file = open(ctag_ctag_file_name, "wb")
    pickle.dump(ctag_ctag_dict, ctag_ctag_file)
    ctag_ctag_file.close()
    print("ctag link records count before filterage: %d" % len(ctag_ctag))

    '''
    ################################
    * 概念-概念标签关系储存导入图数据库 *
    ################################
    '''
    # 过滤同链标签（保留相邻标签节点）后再存作ctag_relations
    ctag_position = pickle.load(open("../Data/Output/Tag_graph/ctag_position.pkl", "rb"))
    label_chains_all = pd.DataFrame.from_dict(ctag_position, orient="index").reset_index()
    label_chains_all.columns = ["node_link", "distance"]
    label_chains_drop = label_chains_all[label_chains_all.distance != 1]
    node_chains = label_chains_all[label_chains_all.distance == 1]
    ctag_ctag_relations = ctag_ctag[~ctag_ctag.tag_link.isin(label_chains_drop.node_link)][["tag_uuid_x", "tag_uuid_y", "link_value", "tag_link"]].copy()
    ctag_ctag_relations["rel_type"] = "ctag_ctag"
    ctag_ctag_relations.loc[ctag_ctag_relations.tag_link.isin(node_chains.node_link), "rel_type"] = "node_of"
    ctag_ctag_relations.drop(["tag_link"], axis=1, inplace=True)
    ctag_ctag_relations.columns = header_dict.get("relation")
    print("ctag link records count after filterage: %d" % len(ctag_ctag_relations))
    return ctag_ctag_relations

# 概念标签和非概念标签关系计算
def ctag_nctag_relation(ctag_comps_aggregated, nctag_comps_aggregated, nctag_idf):
    ctag_comps_aggregated["key"] = 1
    nctag_comps_aggregated["key"] = 1
    nctag_comps_aggregated = nctag_comps_aggregated.merge(nctag_idf, how="left", left_on="tag_uuid", right_on="tag_uuid")
    ctag_nctag = nctag_comps_aggregated.merge(ctag_comps_aggregated, on="key")
    ctag_nctag["link_value"] = ctag_nctag[["comp_int_id_x", "comp_int_id_y", "idf"]].apply(lambda x: x[2]*final_count(x[0], x[1]), axis=1)
    ctag_nctag = ctag_nctag[ctag_nctag.link_value != 0].copy()
    ctag_nctag.link_value = ctag_nctag.link_value.apply(lambda x: np.log2(min(0.000001 + x, 1)))
    ctag_nctag.link_value = simple_minmax(ctag_nctag.link_value)
    ctag_nctag = ctag_nctag[["tag_uuid_x", "tag_uuid_y", "link_value"]].drop_duplicates().copy()
    
    # 概念-非概念标签关联储存作为推荐系统读取使用
    ctag_nctag["tag_link"] = ctag_nctag.tag_uuid_x + "-" + ctag_nctag.tag_uuid_y
    ctag_nctag_dict = dict(zip(ctag_nctag.tag_link, ctag_nctag.link_value))
    ctag_nctag_file_name = "../Data/Output/Tag_graph/ctag_nctag.pkl"
    ctag_nctag_file = open(ctag_nctag_file_name, "wb")
    pickle.dump(ctag_nctag_dict, ctag_nctag_file)
    ctag_nctag_file.close()
    
    '''
    ################################
    * 概念-概念标签关系储存导入图数据库 *
    ################################
    '''
    ctag_nctag_relations = ctag_nctag[["tag_uuid_x", "tag_uuid_y", "link_value"]].copy()
    ctag_nctag_relations["rel_type"] = "ctag_nctag"
    ctag_nctag_relations.columns = header_dict.get("relation")
    print("ctag-nctag link records count: %d" % len(ctag_nctag_relations))
    return ctag_nctag_relations

# 非概念标签两两关系计算
def nctag_nctag(nctag_comps_aggregated, nctag_idf):
    nctag_comps_aggregated["key"] = 1
    nctag_comps_aggregated = nctag_comps_aggregated.merge(nctag_idf, how="left", left_on="tag_uuid", right_on="tag_uuid")
    nctag_nctag = nctag_comps_aggregated.merge(nctag_comps_aggregated, on="key")
    record_len = len(nctag_nctag)
    interval_size = record_len // 20
    i = 0
    # 由于非概念标签之间的计算量级较大，服务器内存不足，因此采用分块计算、最后融合的方式
    while interval_size*i < record_len:
        start_time = datetime.datetime.now()
        print("### start nctag-nctag part %d at %s ###" % (i, start_time.strftime('%H:%M:%S')))
        tmp = nctag_nctag[interval_size*i:min(interval_size * (i + 1), record_len)].copy()
        tmp["link_value"] = tmp[["comp_int_id_x", "comp_int_id_y", "idf_x", "idf_y"]].apply(lambda x: x[2] * x[3] * final_count(x[0], x[1]), axis=1)
        result_part = tmp[tmp.link_value != 0][["tag_uuid_x", "tag_uuid_y", "link_value"]]
        result_part.to_csv("../Data/Output/Tag_graph/temp_result/part_result_%d.relations" % i, index=False, header=None)
        end_time = datetime.datetime.now()
        print("### Part %d finished at %s (time used: %.3f seconds) ###" % (i, end_time.strftime('%H:%M:%S'), (end_time - start_time).total_seconds()))
        i += 1
    print("nctag-nctag calculation done")
    os.system("cat ../Data/Output/Tag_graph/temp_result/part_result_* > ../Data/Output/Tag_graph/temp_result/nctag_nctag_result_all")
    nctag_nctag = pd.read_csv("../Data/Output/Tag_graph/temp_result/nctag_nctag_result_all", header=None)
    nctag_nctag.columns = ["tag1", "tag2", "link_value"]
    nctag_nctag.link_value = nctag_nctag.link_value.apply(lambda x: np.log2(min(0.000001 + x, 1)))
    nctag_nctag.link_value = simple_minmax(nctag_nctag.link_value)
    nctag_nctag = nctag_nctag[["tag1", "tag2", "link_value"]].drop_duplicates().copy()
    nctag_nctag["tag_link"] = nctag_nctag.tag1 + "-" + nctag_nctag.tag2
    nctag_nctag_dict = dict(zip(nctag_nctag.tag_link, nctag_nctag.link_value))
    nctag_nctag_file_name = "../Data/Output/Tag_graph/nctag_nctag.pkl"
    nctag_nctag_file = open(nctag_nctag_file_name, "wb")
    pickle.dump(nctag_nctag_dict, nctag_nctag_file)
    nctag_nctag_file.close()
    
    '''
    ################################
    * 非概念-非概念标签关系导入图数据库 *
    ################################
    '''
    nctag_nctag_relations =  nctag_nctag[["tag1", "tag2", "link_value"]].copy()
    nctag_nctag_relations["rel_type"] = "nctag_nctag"
    nctag_nctag_relations.columns = header_dict.get("relation")
    print("nctag-nctag link records count: %d" % len(nctag_nctag_relations))
    return nctag_nctag_relations


def concept_tree_property(comp_ctag_table_all_infos):   
    tag_code_dict = pd.DataFrame.from_dict(pickle.load(open("../Data/Output/Tag_graph/tag_dict.pkl", "rb")), orient="index").reset_index()
    tag_code_dict.columns = ["label_name", "tag_code"]
    com_ctag_with_code = comp_ctag_table_all_infos.merge(tag_code_dict, how="left", left_on="label_name", right_on="label_name")
    # 每个公司的概念最底层概念标签列表：判断上下游、同链条、同标签
    comp_bottom_ctag = com_ctag_with_code.groupby(["comp_id", "label_type_num"]).apply(lambda x: x[x.label_type == x.label_type.max()])[["comp_id", "tag_code", "comp_full_name"]].drop_duplicates().reset_index(drop=True)
    comp_bottom_ctag.columns = ["comp_id", "bottom_ctag", "comp_full_name"]
    comp_bottom_ctag = comp_bottom_ctag.groupby("comp_id").agg({"bottom_ctag": lambda x: set(x)}).reset_index()
    # 每个公司的顶级概念标签列表：判断同产业“树”
    comp_top_ctag = com_ctag_with_code.groupby(["comp_id", "label_type_num"]).apply(lambda x: x[x.label_type == x.label_type.min()])[["comp_id", "tag_code", "comp_full_name"]].drop_duplicates().reset_index(drop=True)
    comp_top_ctag.columns = ["comp_id", "top_ctag", "comp_full_name"]
    comp_top_ctag = comp_top_ctag.groupby("comp_id").agg({"top_ctag": lambda x: set(x)}).reset_index()
    concept_tree_property = comp_top_ctag.merge(comp_bottom_ctag, how="left", left_on="comp_id", right_on="comp_id")
    concept_tree_property.index = concept_tree_property.comp_id
    concept_tree_property.drop(["comp_id"], axis=1, inplace=True)
    concept_tree_property_dict = concept_tree_property.to_dict(orient='index')
    concept_tree_property_dict_file_name = "../Data/Output/Tag_graph/concept_tree_property.pkl"
    concept_tree_property_dict_file = open(concept_tree_property_dict_file_name, "wb")
    pickle.dump(concept_tree_property_dict, concept_tree_property_dict_file)
    concept_tree_property_dict_file.close()
    return concept_tree_property


# 需要导入neo4j的数据进行合并
def neo4j_merge(tag_points, company_points, comp_tag_relations, ctag_ctag_relations, ctag_nctag_relations, nctag_nctag_relations):
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
    all_relations = pd.concat([ctag_ctag_relations, ctag_nctag_relations, nctag_nctag_relations, comp_tag_relations]) \
        .drop_duplicates().reset_index(drop=True)
    all_relations.columns = header_dict["relation"]
    all_relations["link"] = all_relations[["src_id", "target_id"]].apply(lambda x: "-".join(sorted([str(x[0]), str(x[1])])), axis=1)
    all_relations.drop_duplicates(subset=["link"], inplace=True)
    all_relations.drop(["link"], axis=1, inplace=True)
    all_relations = all_relations[["src_id", "target_id", "rel_value", "rel_type"]].copy()
    all_relations.drop(index=all_relations[all_relations.src_id == all_relations.target_id].index)
    all_relations.reset_index(drop=True, inplace=True)
    all_relations.to_csv("../Data/Output/Tag_graph/all_relations.csv", index=False, header=None)
    print("Relations saved!")
    return 0

def to_graph_database(method="replace"):
    return 0