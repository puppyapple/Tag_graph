from Code import data_generator

comp_tag, company_points, label_chains_raw, comp_ctag_table_all_infos = data_generator.comp_tag()
tag_comps_aggregated, tag_points, comp_tag_relations, comp_total_num = data_generator.data_aggregator(comp_tag, label_chains_raw)
tag_tag_relations = data_generator.tag_tag(tag_comps_aggregated)
data_generator.neo4j_merge(tag_points, company_points, ctag_ctag_relations, ctag_nctag_relations, nctag_nctag_relations)