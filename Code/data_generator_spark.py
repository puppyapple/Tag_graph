# -*- coding: utf-8 -*-
#%%

import uuid
import pickle
import configparser
import pyspark.sql.functions as F
# import gc
from pyspark import SQLContext, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import MinMaxScaler
from pyspark.sql.functions import udf
sc = SparkContext.getOrCreate()
sqlContext=SQLContext(sc)
spark = SparkSession.builder.enableHiveSupport().getOrCreate()


hdfs = "hdfs://10.64.14.69:9000/wuzijun/"
url = "jdbc:mysql://172.31.215.36/innotree_data_assessment"
properties = {"user": "wuzijun","password": "rt6rcU6a11a7on6m"}
config.read(hdfs + "Data/Input/Tag_graph/filter.conf")
ctag_keep_list = config['FILTER']['filter_list'].split(",")

comp_id_name_table = "wuzj_spark_comp_id_name"
tag_uuid_table = "wuzj_spark_tag_uuid"
points_table = "wuzj_spark_points"
relations_table = "wuzj_spark_relations"

# 基本数据处理及生成
def comp_tag(new_result="company_tag_info_latest0703", old_result="company_tag", label_code_relation="label_code_relation", keep_list=False):
    # 从库中读取数据
    sql_new_result = "(select comp_id, comp_full_name, label_name, classify_id, label_type, label_type_num from %s) t" % new_result
    sql_old_result = "(select comp_id, comp_full_name, key_word from %s) t" % old_result
    sql_label_code_relation = "(select * from %s) t" % label_code_relation
     
    data_raw_new = spark.sql(sql_new_result).dropna(how="any")
    data_raw_old_full = sqlContext.read.jdbc(url, sql_old_result, properties).dropna(how="any").withColumnRenamed("key_word", "label_name")
    label_chains_raw = sqlContext.read.jdbc(url, sql_label_code_relation, properties).dropna(how="any")
    label_chains_raw.persist()
    data_raw_new.persist()
    # 生成公司id-name字典保存
    comp_id_name = data_raw_new[["comp_id", "comp_full_name"]].union(data_raw_old_full[["comp_id", "comp_full_name"]]).drop_duplicates()
    comp_id_name.write.jdbc(url=url, table=comp_id_name_table, mode="append")  
    print("comp_id_name_dict saved!")
    
    '''
    ###############################
    * 公司id-名称储存用作导入图数据库 *
    ###############################
    '''
    company_points = comp_id_name[["comp_id", "comp_full_name"]].drop_duplicates(subset=["comp_id"]) \
        .withColumn("property", F.lit("")) \
        .withColumn("point_type", F.lit("company"))

    
    # 全部概念标签的列表
    ctag_full_list = set(label_chains_raw.select("label_note_name") \
                         .rdd.map(lambda x: x[0]).collect() + label_chains_raw.select("label_root_name").rdd.map(lambda x: x[0]).collect())
    '''
    if keep_list == True:
        keep_list = set(ctag_keep_list).union(set(label_chains_raw[label_chains_raw.label_root_name.isin(ctag_keep_list)] \
                                                  .select("label_note_name").rdd.map(lambda x: x[0].collect())))
    else:
        keep_list = ctag_full_list
    '''
    
    comp_ctag_table = data_raw_new[data_raw_new.classify_id != 4][["comp_id", "label_name"]].withColumn("type", F.lit(0))
    
    # 新系统下结果中的公司-非概念标签
    data_raw_nctag_p1 = data_raw_new[data_raw_new.classify_id == 4][["comp_id", "label_name"]]

    # 读取旧版数据，只取其中的非概念标签（概念标签无法确定其层级和产业链（复用））
    data_raw_nctag_p2_raw = data_raw_old_full[data_raw_old_full.key_word != ""][["comp_id", "label_name"]]

    # 新版的非概念标签和旧版整体数据拼接后进行split和flatten
    data_raw_nctag_p2 = data_raw_nctag_p2_raw \
        .rdd.flatMap(lambda x: [(x[0], t) for t in x[1].split(",") if t != ""]).toDF(["comp_id", "label_name"]).drop_duplicates()

    # 取没有概念标记的作为非概念标签的全集
    comp_nctag_table = data_raw_nctag_p1.union(data_raw_nctag_p2)
    comp_nctag_table = comp_nctag_table[~comp_nctag_table.label_name.isin(ctag_full_list)].withColumn("type", F.lit(1))
    comp_tag = comp_ctag_table.union(comp_nctag_table)
    # data_raw_new.unpersist()
    return (comp_tag, label_chains_raw, company_points)


def data_aggregator(comp_tag, nctag_filter_num=(150, 1000000)):
    # 为每一个公司赋予一个整数ID，以减小之后的计算量
    comp_tag = comp_tag.withColumn("comp_int_id", F.monotonically_increasing_id()) \
        .withColumn("count_comps", F.lit(1)) 
    # comp_tag_rdd = comp_tag.rdd.map(lambda x: ((x[1], x[2]), (x[3], x[4])))
    # 将标签数据各自按照标签id进行聚合
    tag_comps_aggregated = comp_tag.groupby("label_name") \
        .agg(F.collect_set("comp_int_id").alias("comp_int_id"), \
             F.max("type").alias("type"), \
             F.sum("count_comps").alias("count_comps"))
    tag_comps_aggregated = tag_comps_aggregated[((tag_comps_aggregated.count_comps >= nctag_filter_num[0]) \
        & (tag_comps_aggregated.count_comps <= nctag_filter_num[1])) \
        | (tag_comps_aggregated.type == 0)]\
        [["label_name", "comp_int_id", "type"]]

    # 为每一个标签赋予一个UUID，这个方式下，只要NAMESPACE不变，重复生成的也会是同一个UUID，避免了增量更新的麻烦
    
    uuid_udf = udf(lambda x: uuid.uuid5(uuid.NAMESPACE_URL, x).hex)
    tag_comps_aggregated = tag_comps_aggregated.withColumn("tag_uuid", uuid_udf("label_name"))
    tag_uuid = tag_comps_aggregated[["label_name", "tag_uuid"]]
    comp_tag = comp_tag.join(tag_uuid, "label_name", "left_outer")
    comp_tag.persist()
    tag_comps_aggregated.persist()
    '''
    ###############################
    * 标签id-名称储存用作导入图数据库 *
    ###############################
    '''
    tag_points = comp_tag[["tag_uuid", "label_name", "type"]]
    tag_type_udf = udf(lambda x: "ctag" if x == 0 else "nctag")
    tag_points = tag_points.withColumn("type", tag_type_udf("type")) \
        .withColumnRenamed("type", "property") \
        .withColumn("point_type", F.lit("tag"))
    tag_points.write.jdbc(url=url, table=points_table, mode="append")
    print("tag points saved!")
    
    '''
    ###############################
    * 公司与标签关系储存用作入图数据库 *
    ###############################
    '''
    comp_tag_relations = comp_tag[["comp_id", "tag_uuid"]].withColumn("rel_value", F.lit(1.0)) \
        .withColumn("rel_type", F.lit("company_tag"))
    comp_tag_relations.write.jdbc(url=url, table=relations_table, mode="append")
    print("company tag link saved!")
    # comp_tag.drop(["label_name", "comp_id"], axis=1, inplace=True)

    tag_uuid.write.jdbc(url=url, table=tag_uuid, mode="append")
    print("tag uuid saved!")
    
    return (comp_tag, tag_comps_aggregated, tag_uuid)

def properties(comp_tag, label_chains_raw, tag_uuid):
    comp_tags_aggregated = comp_tag.groupby("comp_id") \
        .agg(F.collect_set("tag_uuid").alias("comp_tag_list")).dropna(how="any")
    comp_tags_aggregated.write.jdbc(url=url, table=spark_comp_infos_table, mode="append")
    
    # 储存概念标签的位置关系之后作为筛选属性
    ctag_position_file_name = "../Data/Output/Tag_graph/ctag_position.pkl"
    label_chains_raw = label_chains_raw.withColumnRenamed("label_note_name", "label_node_name") \
        .withColumnRenamed("label_type_note":"label_type_node")
    tag_uuid_root = tag_code_dict.withColumnRenamed("tag_uuid": "root_uuid") \
        .withCOlumnRenamed("label_name":"root_name")
    tag_uuid_node = tag_code_dict.rename(index=str, columns={"tag_code":"node_code", "label_name":"node_name"}, inplace=False)
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



# 标签两两关系计算2
def tag_tag2():
    tag_comps_aggregated_raw = pickle.load(open("../Data/Output/Tag_graph/tag_comps_aggregated.pkl", "rb"))
    tag_comps_aggregated_raw.comp_int_id = tag_comps_aggregated_raw.comp_int_id.apply(lambda x: list(x))
    tag_comps_aggregated = sqlContext.createDataFrame(tag_comps_aggregated_raw)
    result = tag_comps_aggregated.withColumnRenamed("comp_int_id", "comp_int_id2") \
        .withColumnRenamed("type", "type2") \
        .withColumnRenamed("tag_uuid", "tag_uuid2") \
        .crossJoin(tag_comps_aggregated) \
        .rdd.map(lambda x: (x[2], x[5]) + (x[1] + x[4], final_count(set(x[0]), set(x[3])))) \
        .distinct().toDF(["tag1", "tag2", "link_type", "link_value"]).filter("link_value > 0")
    tag_tag = result.toPandas()
    tag_tag.columns = ["tag1", "tag2", "link_type", "link_value"]
    tag_tag.link_value = nctag_nctag.link_value.apply(lambda x: np.log2(min(0.000001 + x, 1)))
    tag_tag.link_value = simple_minmax(nctag_nctag.link_value)
    tag_tag = tag_tag[["tag1", "tag2", "link_type", "link_value"]].drop_duplicates()
    tag_tag["tag_link"] = tag_tag.tag1 + "-" + tag_tag.tag2
    tag_tag_dict = dict(zip(tag_tag.tag_link, (tag_tag.link_value, tag_tag.link_type)))
    tag_tag_file_name = "../Data/Output/Tag_graph/tag_tag.pkl"
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
    tag_tag_relations.to_csv("../Data/Output/Tag_graph/tag_tag_relations.relations")
    
    print("tag-tag link records count: %d" % len(tag_tag_relations))
    del(tag_comps_aggregated)
    del(tag_tag)
    gc.collect()
    return 0


# 需要导入neo4j的数据进行合并
def neo4j_merge(tag_points, company_points, comp_tag_relations, tag_tag_relations):
    # 全部点集合
    tag_points = pd.read_csv("../Data/Output/Tag_graph/tag_points.points")
    company_points = pd.read_csv("../Data/Output/Tag_graph/company_points.points")
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
    comp_tag_relations = pd.read_csv("../Data/Output/Tag_graph/comp_tag_relations.relations")
    tag_tag_relations = pd.read_csv("../Data/Output/Tag_graph/tag_tag_relations.relations")
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

