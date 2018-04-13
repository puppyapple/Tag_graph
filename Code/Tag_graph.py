import pandas as pd
import numpy as np 

data_raw = pd.read_csv("./Data/company_tag_data_raw", sep='\t')
print(len(data_raw))
