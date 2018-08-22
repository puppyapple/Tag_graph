from Code import data_generator_new
import pickle
import pandas as pd
import sys

version = "v1"
path_comp_tags_all = "../Data/Output/Tag_graph/%s/comp_tags_all.pkl" % version
path_tag_tag = "../Data/Output/Tag_graph/%s/tag_tag.pkl" % version
path_comp_id_name = "../Data/Output/Tag_graph/%s/all_points.csv" % version
path_tag_dict = "../Data/Output/Tag_graph/%s/tag_dict.pkl" % version
path_comp_tags_all_sparse = "../Data/Output/Tag_graph/%s/comp_tags_all_sparse.pkl" % version
path_tag_tag_sparse = "../Data/Output/Tag_graph/%s/tag_tag_no_log_sparse.pkl" % version


def data_producer(new_table_name):
    data_generator_new.comp_tag(new_table_name)
    data_generator_new.data_aggregator()
    data_generator_new.properties()
    data_generator_new.tag_tag4()
    dense_to_sparse()
    data_generator_new.neo4j_merge()
    data_generator_new.to_graph_database()

def data_loader():
    comp_tags_all = pickle.load(open(path_comp_tags_all, "rb"))
    tag_tag = pickle.load(open(path_tag_tag, "rb"))
    comp_id_name = pd.read_csv(path_comp_id_name, header=None, dtype={0: str})
    comp_id_name.columns = ["point_id", "name", "property", "point_type"]
    comp_id_name.dropna(subset=["point_id", "name"], inplace=True)
    comp_id_name.name = comp_id_name.name.apply(lambda x: x.strip())
    tag_dict = pickle.load(open(path_tag_dict, "rb"))
    comp_tags_all_sparse = pickle.load(open(path_comp_tags_all_sparse, "rb"))
    tag_tag_sparse = pickle.load(open(path_tag_tag_sparse, "rb"))
    return (comp_tags_all, tag_tag, comp_id_name, tag_dict, comp_tags_all_sparse, tag_tag_sparse)

    
if __name__ == '__main__':
    table_name = sys.argv[1]
    data_producer(table_name)