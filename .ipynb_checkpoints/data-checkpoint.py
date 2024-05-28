import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

df_iq = pd.DataFrame(pd.read_csv('iq.csv'))
df_qol = pd.DataFrame(pd.read_csv('quality_of_life.csv'))
PATH = Path('C:/Users/jk101/Desktop/python/data/data.csv')

output1 = pd.merge(df_iq, df_qol,  
                   on='country',  
                   how='left')

for r in range(output1.shape[0]):
    for c in output1.columns:
        print(output1.at[r, str(c)])

output1.to_csv(PATH)