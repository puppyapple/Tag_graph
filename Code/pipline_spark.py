import data_generator_new
import pickle

version = "v1"
path_comp_tags_all = "../Data/Output/Tag_graph/%s/comp_tags_all.pkl" % version
path_tag_tag = "../Data/Output/Tag_graph/%s/tag_tag.pkl" % version
path_comp_id_name_dict = "../Data/Output/Tag_graph/%s/comp_id_name_dict.pkl" % version
path_tag_dict = "../Data/Output/Tag_graph/%s/tag_dict.pkl" % version

def data_producer():
    comp_tag, label_chains_raw = data_generator_new.comp_tag()
    data_generator_new.data_aggregator()
    data_generator_new.properties()
    tag_tag_relations = data_generator_new.tag_tag()
    # data_generator_new.neo4j_merge()

def data_loader():
    comp_tags_all = pickle.load(open(path_comp_tags_all, "rb"))
    tag_tag = pickle.load(open(path_tag_tag, "rb"))
    comp_id_name_dict = pickle.load(open(path_comp_id_name_dict, "rb"))
    tag_dict = pickle.load(open(path_tag_dict, "rb"))
    return (comp_tags_all, tag_tag, comp_id_name_dict, tag_dict)