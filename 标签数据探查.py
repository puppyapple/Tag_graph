
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import pyspark
import operator
import matplotlib.pyplot as plt
from pyspark import SparkContext 
from pyspark.sql import SQLContext 
from collections import Counter
from itertools import groupby
from functools import reduce
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


file_name = "company_tag_data_v56"


# In[4]:


data_raw = pd.read_csv("../Data/Input/" + file_name, sep='\t', dtype={"comp_id":str})[["comp_id", "comp_full_name", "label_name", "classify_id", "label_type", "label_type_num", "src_tags"]]


# In[5]:


data_raw.comp_full_name = data_raw.comp_full_name.apply(lambda x: x.strip().replace("(","（").replace(")","）"))


# ## 四个类别各自包含的标签（去重）数量及占比

# In[6]:


label_name_by_class = data_raw[['classify_id', 'label_name']].drop_duplicates().groupby('classify_id').count()
label_name_by_class.plot(kind='bar')
label_name_by_class.reset_index(inplace=True)
label_name_by_class['percentage'] = (label_name_by_class.label_name/label_name_by_class.label_name.sum()).apply(lambda x: '%.2f%%' % (x * 100))
label_name_by_class


# ## 各个标签各自包含的公司（去重）数目及占比

# In[7]:


comp_by_class = data_raw[['classify_id', 'comp_full_name']].drop_duplicates().groupby('classify_id').count()
comp_by_class.plot(kind='bar')
comp_by_class.reset_index(inplace=True)
comp_by_class['percentage'] = (comp_by_class.comp_full_name/comp_by_class.comp_full_name.sum()).apply(lambda x: '%.2f%%' % (x * 100))
comp_by_class


# In[8]:


data_raw


# In[9]:


non_concept_raw = data_raw[data_raw.classify_id == 4][['label_name', 'comp_id']].drop_duplicates()
non_concept_raw.columns = ['non_concept_label', 'comp_id']
non_concept_raw


# In[10]:


concept = data_raw[data_raw.classify_id != 4][['label_name', 'comp_id']].drop_duplicates()
concept.columns = ['concept_label', 'comp_id']


# In[11]:


sc = SparkContext.getOrCreate()
sqlContext=SQLContext(sc)


# In[12]:


non_concept_df = sqlContext.createDataFrame(non_concept_raw)
non_concept = sqlContext.createDataFrame(non_concept_df.rdd.flatMap(lambda x: map(lambda y: (y, x[1]), x[0].split(','))), schema=['non_concept_label', 'comp_id']).toPandas()


# In[13]:


label_relation = non_concept.merge(concept, how='left', left_on='comp_id', right_on='comp_id').fillna('')
label_relation['pair_count'] = label_relation.concept_label.apply(lambda x: 0 if x =='' else 1)


# In[14]:


label_relation


# ## 以非概念标签为单位，与之共现的概念标签次数及列表

# In[15]:


relation_count = label_relation.groupby('non_concept_label').agg({'concept_label': lambda x: Counter([w for w in x if w != '']), 'pair_count': 'sum'}).reset_index()
relation_count.concept_label = relation_count.concept_label.apply(lambda x: str(sorted(dict(x).items(), key=operator.itemgetter(1),reverse=True)).replace('[', '').replace(']', '').replace("'", ''))


# In[16]:


relation_count.to_excel('../Data/Output/non_concept_words_statistics.xlsx')


# In[17]:


label_relation2 = concept.merge(non_concept, how='left', left_on='comp_id', right_on='comp_id').fillna('')
label_relation2['pair_count'] = label_relation2.concept_label.apply(lambda x: 0 if x == '' else 1)


# ## 以概念标签为单位，与之共现的非概念标签次数及列表

# In[18]:


relation_count2 = label_relation2.groupby('concept_label').agg({'non_concept_label': lambda x: Counter([w for w in x if w != '']), 'pair_count': 'sum'}).reset_index()
relation_count2.non_concept_label = relation_count2.non_concept_label.apply(lambda x: str(sorted(dict(x).items(), key=operator.itemgetter(1),reverse=True)).replace('[', '').replace(']', '').replace("'", ''))


# In[19]:


relation_count2.to_excel('../Data/Output/concept_words_statistics.xlsx')


# In[20]:


print("非概念标签中与概念标签完全无交集公司的个数为：%s" % sum(relation_count.pair_count == 0))
print("非概念标签中与概念标签有交集公司的个数为：%s" % sum(relation_count.pair_count != 0))


# In[21]:


relation_count.pair_count.describe()


# In[22]:


lst = [c if c<=150 else 150 for c in list(relation_count.pair_count) if c != 0]
dic = {}
for k, g in groupby(sorted(lst), key=lambda x: x//5):
    dic['{}-{}'.format(k*5+1, (k+1)*5)] = len(list(g))


# In[23]:


interval_cnt = pd.DataFrame.from_dict(dic, 'index').reset_index()
interval_cnt.columns = ['interval', 'count_of_related_concept_tag']
interval_cnt.plot('interval', 'count_of_related_concept_tag',kind='bar')
interval_cnt


# In[24]:


a = non_concept[['comp_id']].drop_duplicates().merge(concept, how='left', left_on='comp_id', right_on='comp_id')
len(a[a.concept_label.isnull()])


# In[25]:


len(set(data_raw[data_raw.classify_id !=4]['comp_id']))


# In[26]:


len(set(data_raw.comp_id))


# In[27]:


data_new = data_raw.copy()
data_new.classify_id = data_new.classify_id.apply(lambda x: 'nc' if x == 4 else 'c')


# In[28]:


count_two_labels = data_new.pivot_table('label_name', index='comp_id', columns='classify_id', aggfunc='count').fillna(0.0)


# In[29]:


count_two_labels['only_c'] = (count_two_labels.c != 0) & (count_two_labels.nc == 0)
count_two_labels['only_nc'] = (count_two_labels.c == 0) & (count_two_labels.nc != 0)
count_two_labels['n&c'] = (count_two_labels.c != 0) & (count_two_labels.nc != 0)
count_two_labels['none'] = (count_two_labels.c == 0) & (count_two_labels.nc == 0)
pd.DataFrame(count_two_labels.sum())


# In[30]:


pie = count_two_labels[['only_c', 'only_nc', 'n&c']].sum()/len(set(count_two_labels.index))
labels = ['only_concept', 'only_non-concept', 'concept&non-concept']
explode = [0,0.1,0]
len(set(count_two_labels.index))


# In[31]:


plt.axes(aspect='equal')
plt.pie(pie, explode=explode, labels=labels, autopct='%.2f%%', radius = 1.5, wedgeprops = {'linewidth': 1.5, 'edgecolor':'black'},  textprops = {'fontsize':16, 'color':'k'})


# In[32]:


len(set(data_raw[data_raw.classify_id!=4].label_name))


# In[33]:


label_relation3 = label_relation[label_relation.pair_count != 0]


# In[34]:


non_concept_label_vector = label_relation3.groupby('non_concept_label').agg({'concept_label': lambda x: dict(Counter([w for w in x if w != ''])), 'pair_count': 'sum'}).reset_index()


# In[35]:


non_concept_label_vector


# In[42]:


concept_label_dict_raw = data_raw[data_raw.classify_id != 4]['label_name'].drop_duplicates().reset_index(drop=True)
concept_label_dict = dict(concept_label_dict_raw)
vector_len = len(concept_label_dict)
company_num = len(set(data_raw.comp_id))
company_num


# In[44]:


concept_label_list = list(concept_label_dict.values())
non_concept_label_vector['concept_label_vector'] = non_concept_label_vector.concept_label.apply(lambda x: np.array(list(map(lambda y: x.get(y, 0) ,concept_label_list))))
non_concept_label_vector


# In[38]:


type(non_concept_label_vector.concept_label_vector[0])


# In[46]:


comp_count_by_nc = non_concept.groupby('non_concept_label').count().reset_index()
comp_count_by_nc['nc_label_weight'] = np.log(company_num/(comp_count_by_nc.comp_id + 1))
nc_vector_weight = non_concept_label_vector[['non_concept_label', 'concept_label_vector']].merge(comp_count_by_nc[['non_concept_label', 'nc_label_weight']], how='left', left_on='non_concept_label', right_on='non_concept_label')
nc_vector_weight['product'] = (nc_vector_weight.concept_label_vector.multiply(nc_vector_weight.nc_label_weight))


# In[47]:


comp_vectors_raw = non_concept.merge(nc_vector_weight, how='left',  left_on='non_concept_label', right_on='non_concept_label').dropna(how='any')
comp_vectors_raw


# In[48]:


comp_by_concept_vector = comp_vectors_raw[['comp_id', 'product']].groupby('comp_id').agg('sum').reset_index()


# In[92]:


top_n = 5
comp_by_concept_vector['sorted_vector_index'] = comp_by_concept_vector['product'].apply(lambda x: np.argpartition(x, -top_n)[-top_n:])


# In[93]:


comp_by_concept_vector['top_concept_labels'] = comp_by_concept_vector.sorted_vector_index.apply(lambda x: [concept_label_dict.get(i, 'Not found') for i in x])


# In[94]:


comp_with_concept = data_raw[data_raw.classify_id != 4][['comp_id', 'label_name']].groupby('comp_id').agg({'label_name': lambda x: dict(Counter([w for w in x if w != '']))}).reset_index()


# In[70]:


comp_with_concept


# In[95]:


result = comp_by_concept_vector.merge(comp_with_concept, how='left', left_on='comp_id', right_on='comp_id').merge(data_raw[['comp_id', 'comp_full_name']].drop_duplicates(), how='left', left_on='comp_id', right_on='comp_id').fillna('No concept label')
result.rename(index=str, columns={'label_name': 'real_concept_labels'}, inplace=True)
result


# In[96]:


result[['comp_id', 'comp_full_name', 'top_concept_labels', 'real_concept_labels']].to_excel('../Data/Output/concept_vector_test.xlsx')

