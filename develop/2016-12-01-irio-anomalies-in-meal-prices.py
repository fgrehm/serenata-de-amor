
# coding: utf-8

# # Anomalies in meal prices
# 
# In the Chamber of Deputies' CEAP, there is a list of 1,000's of meal expenses made by congresspeople. The law says that the congressperson cannot pay for any other, even being her advisor or SO. We want to work on this analysis to find possibly illegal and immoral expenses. They may have happened when the politician spent more than needed (e.g. the whole menu costs X but the bill was 2X) or too much in an specific period of time. In the end, we also want to alert about too expensive reibursements, even with an explanation behind of it.

# In[1]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

plt.rcParams['figure.figsize'] = (20, 10)


# In[2]:

import numpy as np
import pandas as pd

dataset = pd.read_csv('../data/2016-11-19-reimbursements.xz',
                      dtype={'applicant_id': np.str,
                             'cnpj_cpf': np.str,
                             'congressperson_id': np.str,
                             'subquota_number': np.str},
                      low_memory=False)


# In[3]:

is_in_brazil = '(-73.992222 < longitude < -34.7916667) & (-33.742222 < latitude < 5.2722222)'
companies = pd.read_csv('../data/2016-09-03-companies.xz',
                        dtype={'cnpj': np.str},
                        low_memory=False)
companies = companies.query(is_in_brazil)
companies['cnpj'] = companies['cnpj'].str.replace(r'\D', '')
dataset = pd.merge(dataset, companies,
                   how='left',
                   left_on='cnpj_cpf',
                   right_on='cnpj',
                   suffixes=('', '_company'))


# In[78]:

dataset =     dataset.query('subquota_description == "Congressperson meal"')
companies =     companies[companies['cnpj'].isin(dataset.loc[dataset['cnpj'].notnull(),
                                                 'cnpj'])]


# In[79]:

dataset['total_net_value'].describe()


# In[80]:

dataset['total_net_value'].median()


# In[81]:

top_99 = dataset['total_net_value'].quantile(0.99)
top_99


# In[82]:

dataset[dataset['total_net_value'] < top_99].shape


# In[83]:

sns.distplot(dataset.loc[dataset['total_net_value'] < top_99, 'total_net_value'],
             bins=30,
             kde=False)


# In[84]:

top_99_dataset = dataset.query('total_net_value > {}'.format(top_99))
ranking = top_99_dataset.groupby('state_company')['total_net_value']     .median().sort_values(ascending=False)

sns.boxplot(x='state_company',
            y='total_net_value',
            data=top_99_dataset,
            order=ranking.index)


# In[20]:

top_99_dataset.query('state_company == "CE"').shape


# In[22]:

dataset.query('state_company == "CE"').shape


# In[21]:

top_99_dataset['state_company'].isnull().sum()


# In[18]:

top_99_dataset.query('state_company == "CE"')     .sort_values('total_net_value', ascending=False)


# In[29]:

yelp = pd.read_csv('../data/2016-11-29-yelp-companies.xz',
                   low_memory=False)
yelp.head()


# We have data for just 8.6% of the companies which received from the "Congressperson meal" subquota.

# In[66]:

yelp['price'].notnull().sum()


# In[77]:

companies.shape


# In[30]:

yelp['price'].isnull().sum()


# In[58]:

yelp['price.int'] = yelp['price'].str.len()
states_with_records =     yelp[yelp['price'].notnull()].groupby('location.state')['location.state'].count() > 10
states_with_records = states_with_records[states_with_records].index


# In[64]:

yelp_just_significant_states =     yelp[yelp['price'].notnull() &
         yelp['location.state'].isin(states_with_records)]
yelp_just_significant_states['location.state'].value_counts()


# In[65]:

yelp_ranking = yelp_just_significant_states.groupby('location.state')['price.int']     .median().sort_values(ascending=False)
    
sns.swarmplot(x='location.state',
              y='price.int',
              data=yelp_just_significant_states,
              order=yelp_ranking.index)


# In[ ]:



