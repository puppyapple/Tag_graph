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
from pyspark.ml.linalg import Vectors
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
def comp_tag(new_result="spider_clean_sum.company_tag_new", label_code_relation="label_code_relation", keep_list=False):
    # 从库中读取数据
    sql_new_result = "select comp_id, comp_full_name, label_name, classify_id, label_type, label_type_num from %s" % new_result
    sql_label_code_relation = "(select * from %s) t" % label_code_relation
     
    data_raw_new = spark.sql(sql_new_result).dropna(how="any")
    label_chains_raw = sqlContext.read.jdbc(url, sql_label_code_relation, properties).dropna(how="any")
        .withColumnRenamed("label_note_name", "label_node_name") \
        .withColumnRenamed("label_type_note":"label_type_node")
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
    

    # 取没有概念标记的作为非概念标签的全集
    comp_nctag_table = data_raw_new[(data_raw_new.classify_id == 4) & (~data_raw_new.label_name.isin(ctag_full_list))] \
        [["comp_id", "label_name"]] \
        .withColumn("type", F.lit(1))

    comp_tag_raw = comp_ctag_table.union(comp_nctag_table)
    # data_raw_new.unpersist()
    return (comp_tag_raw, label_chains_raw, company_points)


def data_aggregator(comp_tag_raw, nctag_filter_num=(150, 1000000)):
    # 为每一个公司赋予一个整数ID，以减小之后的计算量
    comp_tag = comp_tag_raw.withColumn("comp_int_id", F.monotonically_increasing_id()) \
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
    tag_comps_aggregated = tag_comps_aggregated.withColumn("tag_uuid", uuid_udf("label_name"))[["tag_uuid", "comp_int_id", "type"]]
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
        .withColumn("remark", F.lit(98))
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
    print("comp tags infos saved!)
    
    # 储存概念标签的位置关系之后作为筛选属性
    tag_uuid_root = tag_code_dict.withColumnRenamed("tag_uuid": "root_uuid") \
        .withColumnRenamed("label_name":"root_name")
    tag_uuid_node = tag_code_dict.withColumnRenamed("tag_uuid": "node_uuid") \
        .withColumnRenamed("label_name":"node_name")
    label_chains_raw = label_chains_raw \
        .join(tag_uuid_node, label_chains_raw.label_node_name==tag_uuid_node.node_name, "left_outer") \
        .join(tag_uuid_root, label_chains_raw.label_root_name==tag_uuid_node.root_name, "left_outer")
        .withColumn("distance", label_chains_raw["label_type_node"] - label_chains_raw["label_type_root"])[["node_code", "root_code", "distance"]]

    label_chains_link = label_chains_raw \
        .union(label_chains_raw.withColumn("distance", - label_chains_raw["distance"])[["root_code", "node_code", "distance"]])

    label_self = label_chains_link.select("node_code") \
        .withColumn("root_code", label_chains_link.node_code)
        .withColumn("distance", F.lit(0))
    label_chains_link = label_chains_link.union(label_self) \
        .withColumn("node_link", F.concat_ws("-", label_chains_link["node_code"], label_chains_link["root_code"]))
    return label_chains_link


## 标签关系计算
def final_count(l1, l2):
    if len(set(l1 + l2)) == 0:
        return 0
    else:
        return len(x for x in l1 if x in l2)/len(set(l1 + l2))

def simple_minmax(df, target_col_name, min_v=0.001, max_v=1):
    mms = MinMaxScaler(inputCol=target_col_name, outputCol=target_col_name + "_scaled", min=min_v)
    model = mms.fit(df)
    df = model.transform(df)
    return df


# 标签两两关系计算2
def tag_tag2(tag_comps_aggregated, label_chains_link):
    log_udf = udf(lambda x: math.log(min(0.000001 + x, 1)))
    rel_type_dict = {0: "ctag_ctag", 1: "ctag_nctag", 2: "nctag_nctag"}
    rel_type_udf = udf(lambda x: rel_type_dict.get(x))
    result = tag_comps_aggregated.withColumnRenamed("tag_uuid", "tag_uuid2") \
        .withColumnRenamed("comp_int_id", "comp_int_id2") \
        .withColumnRenamed("type", "type2") \
        .crossJoin(tag_comps_aggregated) \
        .rdd.map(lambda x: (x[0], x[3]) + (final_count(x[1], x[4]), x[2] + x[5])) \
        .distinct() \
        .toDF(["src_id", "target_id", "rel_value", "rel_type"]) \
        .filter("rel_value > 0")
        .withColumn("rel_value", log_udf("rel_value"))
    tag_tag = simple_minmax(result, "rel_value")[["src_id", "target_id", "rel_type", "rel_value_scaled"]] \
        .drop_duplicates() \
        .withColumnRenamed("rel_value_scaled", "rel_value") \
        .withColumn("node_link", F.concat_ws("-", result["src_id"], result["target_id"])) \
        .join(label_chains_link[label_chains_link.distance == 1][["node_link", "distance"]], "node_link", "left_outer") \
        .fillna(99) \
        .withColumn("rel_type", rel_type_udf("rel_type")) \
        .withColumnRenamed("distance", "remark")[["src_id", "target_id", "rel_type", "rel_value", "remark"]] \
        .rdd \
        .map(lambda x: ("-".join(sorted((x[0], x[1]))), x[0], x[1], x[2], x[3], x[4])) \
        .toDF(["sorted_link", "src_id", "target_id", "rel_type", "rel_value", "remark"]) \
        .drop_duplicates(subset=["sorted_link"]) \
        [["src_id", "target_id", "rel_type", "rel_value", "remark"]]
    
    '''
    ################################
    * 概念-概念标签关系用于导入图数据库 *
    ################################
    '''
    tag_tag.write.jdbc(url=url, table=relations_table, mode="append")
    print("tag tag link saved!")
          
    return 0


if __name__ == "__main__":
    comp_tag, label_chains_raw, company_points = comp_tag()
    