from Code import data_generator

comp_ctag_table, comp_nctag_table, company_points, label_chains_raw, comp_ctag_table_all_infos = data_generator.comp_tag()
ctag_comps_aggregated, nctag_comps_aggregated, tag_points, company_points, comp_tag_relations, comp_total_num = data_generator.data_aggregator(comp_ctag_table, comp_nctag_table, company_points, label_chains_raw)
nctag_idf = data_generator.nctag_idf(nctag_comps_aggregated, comp_total_num)
ctag_ctag_relations, ctag_nctag_relations, nctag_nctag_relations = data_generator.ctag_relation(ctag_comps_aggregated), data_generator.ctag_nctag_relation(ctag_comps_aggregated, nctag_comps_aggregated, nctag_idf), data_generator.nctag_nctag(nctag_comps_aggregated, nctag_idf)
data_generator.concept_tree_property(comp_ctag_table_all_infos)
data_generator.neo4j_merge(tag_points, company_points, ctag_ctag_relations, ctag_nctag_relations, nctag_nctag_relations)