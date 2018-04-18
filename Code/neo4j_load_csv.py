#%%
import os
from neo4j.v1 import GraphDatabase

#%%
uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("innotree", "innotree"))

#%%
with driver.session() as session:
    session.run("using periodic commit 1000 LOAD CSV WITH HEADERS FROM 'file:///tags.points' AS line \
        MERGE (t:tag{标签代码:line.tag_code,标签名称:line.tag_name,标签级别:line.tag_level})")
    print("Tags loaded!")

#%%
with driver.session() as session:
    session.run("using periodic commit 5000 LOAD CSV WITH HEADERS FROM 'file:///companies.points' AS line \
        MERGE (c:company{公司代码:line.comp_id,公司全称:line.comp_full_name})")
    print("Companies loaded!")

#%%
with driver.session() as session:
    session.run("using periodic commit 1000 LOAD CSV WITH HEADERS FROM 'file:///level_tag_value.relations' AS line \
         MATCH (from:tag{标签代码:line.node_code}),(to:tag{标签代码:line.root_code}) \
         MERGE (from)-[r:lt_rel{公司占比:line.proportion}]->(to)")
    print("Level tag relations loaded!")

#%%
with driver.session() as session:
    session.run("using periodic commit 1000 LOAD CSV WITH HEADERS FROM 'file:///tag_relation_value.relations' AS line \
         MATCH (from:tag{标签代码:line.tag1}),(to:tag{标签代码:line.tag2}) \
         MERGE (from)-[r:tt_rel{公司交集数:line.intersection,公司并集数:line.union,关联强度:line.percentage}]->(to) ")
    print("Tag tag relations loaded!")

#%%
with driver.session() as session:
    session.run("using periodic commit 1000 LOAD CSV WITH HEADERS FROM 'file:///company_tag.relations' AS line \
         MATCH (from:company{公司代码:line.comp_id}),(to:tag{标签代码:line.tag_code}) \
         MERGE (from)-[r:ct_rel{}]->(to) ")
    print("Company tag relations loaded!")
#%%

