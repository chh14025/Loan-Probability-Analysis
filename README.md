# Loan-Probability-Analysis
E-signing a loan based on Financial History Probability Analysis

__A logistic regression machine learning model that explores the likelyhood of loan applicants that will go through with the final E-signing phase based on their financial data - 63% accuracy__

## EDA
The initiation phase of the project starts with data cleaning and visualization <br />
To do that we first need to import the necessary libraries 

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib qt
```
read the Financial_Data.csv file and get a better idea of what kind of data we are dealing with by using :<br />

dataset = pd.read_csv('Financial_Data.csv') 
dataset.describe()

![](Images/Desc.png)


We then clean up the data a bit by looking for any N/A data in our columns<br />

```python
dataset.isna().any()
```

Fortunate for us, the data-set we have is clean and contains no N/A data.<br />

Moving on, we will generate a list of Histograms of known variables to see if we can find any patterns, <br />
To do this we first need to drop columns containing identifiers or dependent variables. <br />

dataset2 = dataset.drop(columns = ['entry_id', 'pay_schedule', 'e_signed']) <br />

With all the columns preped, we can generate a grid of histograms

```python
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

```

![](Images/Desc.png)

Here find a lot of valuable information, let's break down the variables that stood out:<br />

_Age_ - The majority of loan applicants are bettween the ages of 30 and 50. <br />
_Home Owner_ - There are less home owners than there are none-home owners. <br />
_Income_ - The income of this group of loan applicants lies between 15k to 60k with incriments of ~5k. <br />
_Years Employed_ - Most of the applicants has worked less than 7 years. <br />
_Personal Account(Year)_ - Most of the loan applicants have two or fewer accounts. A small bump in numbers of applicants with 6 accounts.<br />
_Has Debt_ - A very large majority of applicants have debt. <br />
_Amount Requested_ - The amount requested is mostly between 1k to 15k. <br />
_Risk Score_ - The risk scores are in the 60k region. <br />
_Inquires Last Month_ - Inquires within the last month is between 2 to 10. <br />









