import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib qt
dataset = pd.read_csv('Financial_Data.csv') 

dataset.describe()


#Cleaning the data

dataset.isna().any()

dataset2 = dataset.drop(columns = ['entry_id', 'pay_schedule', 'e_signed'])

fig = plt.figure(figsize = (15,12))
plt.suptitle('Histogram of Numerical Columns')
for i in range (dataset2.shape[1]):
    plt.subplot(6,3,i+1)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i])
    
    val = np.size(dataset2.iloc[:,i].unique())
    if val >= 100:
        val = 100
        
    plt.hist(dataset2.iloc[:,i], bins = val)
plt.tight_layout(rect = [0,0.1,1,0.9])
    

dataset2.corrwith(dataset.e_signed).plot.bar(
    title = 'Correlation with E-signed')

#Correlation Matrix
sns.set(style = 'white')

corr = dataset2.corr()

mask = np.zeros_like(corr, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots()

cmap = sns.diverging_palette(220,10,as_cmap = True)

sns.heatmap(corr,mask = mask, cmap = cmap, vmax = 0.3, center =0,
            square = True, linewidths = 0.5, cbar_kws = {"shrink": .5})

