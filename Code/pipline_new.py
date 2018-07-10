from Code import data_generator_new

comp_tag, company_points, label_chains_raw = data_generator_new.comp_tag()
tag_comps_aggregated, tag_points, comp_tag_relations = data_generator_new.data_aggregator(comp_tag, label_chains_raw)
tag_tag_relations = data_generator_new.tag_tag(tag_comps_aggregated)
data_generator_new.neo4j_merge(tag_points, company_points, tag_tag_relations)