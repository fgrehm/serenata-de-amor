
# coding: utf-8

# # Traveled speeds
# 
# The Quota for Exercise of Parliamentary activity says that meal expenses can be reimbursed just for the politician, excluding guests and assistants. Creating a feature with information of traveled speed from last meal can help us detect anomalies compared to other expenses.
# 
# * Learn how to calculate distance between two coordinates.
# * Filter "Congressperson meal" expenses.
# * Order by occurence.
# * Merge `current-year.xz` datasets with `companies.xz`, so we have latitude/longitude for each expense.
# * Remove expenses with less than 12 hours of distance between each other.
# 
# 
# * Filter specific congressperson.
# 
# ...

# In[1]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def read_csv(name='last-year'):
    data = pd.read_csv('../data/2016-08-08-%s.xz' % name,
                       parse_dates=[16],
                       dtype={'document_id': np.str,
                              'congressperson_id': np.str,
                              'congressperson_document': np.str,
                              'term_id': np.str,
                              'cnpj_cpf': np.str,
                              'reimbursement_number': np.str})
    congresspeople = pd.read_excel('~/Downloads/deputado.xls')
    is_individual_document = data['congressperson_name'].isin(congresspeople['Nome Parlamentar'])
    is_meal_document = data['subquota_description'] == 'Congressperson meal'
    return data[is_individual_document & is_meal_document]

def document_url(record):
    return 'http://www.camara.gov.br/cota-parlamentar/documentos/publ/%s/%s/%s.pdf' %         (record['applicant_id'], record['year'], record['document_id'])


datasets = [read_csv(name)
            for name in ['current-year', 'last-year', 'previous-years']]
data = pd.concat(datasets)
del(datasets)
data['issue_date'] = pd.to_datetime(data['issue_date'])
data = data.sort_values('issue_date')


# In[2]:

len(data['congressperson_name'].unique())


# In[3]:

data.groupby('congressperson_name', as_index=False).     agg({'net_value': np.nansum}).     sort_values('net_value', ascending=False).     head(10)
# congressperson_list = data. \
#     drop_duplicates('applicant_id', keep='first')
# ranking = pd.merge(data,
#                    congressperson_list,
#                    how='left',
#                    on='applicant_id')
# ranking.head(10)


# In[4]:

document = data.sample(random_state=0).iloc[0]
document


# In[5]:

print(document_url(document))


# In[6]:

# ID | speed_from_previous | speed_from_next
# ------------------------------------------
# B  | None                | 
# A  | 
# C  | 


# In[7]:

import numpy as np

aggregation = data.     groupby(['congressperson_name', 'issue_date'])['net_value'].     agg([len, np.sum, np.nanmean]).     reset_index()
aggregation.sort_values(['nanmean', 'len'], ascending=[False, False])


# In[8]:

sns.lmplot('len', 'sum',
           data=aggregation,
           fit_reg=False,
           scatter_kws={"marker": "D", 
                        "s": 100})


# In[9]:

aggregation[aggregation['len'] > 8]


# In[10]:

magda = data[(data['congressperson_name'] == 'MAGDA MOFATTO') & (data['issue_date'] == '2015-03-15')]
magda = magda[magda['document_id'].isin(['5645173', '5645177'])]
document_urls = magda.     apply(lambda row: pd.Series({'document_url': document_url(row)}), axis=1)

equal = pd.concat([magda, document_urls], axis=1)
equal


# In[11]:

record = equal.iloc[0]
print(record['document_url'])
record


# In[12]:

record = equal.iloc[1]
print(record['document_url'])
record


# ## Add distance from last day

# In[13]:

companies = pd.read_csv('../data/2016-09-03-companies.xz', low_memory=False)
companies['cnpj'] = companies['cnpj'].str.replace(r'[./-]', '')


# In[14]:

list(companies.columns[:26]) + ['latitude', 'longitude']


# In[15]:

companies = companies[list(companies.columns[:26]) + ['latitude', 'longitude']]


# In[16]:

data_with_geo = pd.merge(data, companies,
                         how='left',
                         left_on='cnpj_cpf',
                         right_on='cnpj')
data_with_geo = data_with_geo[data_with_geo['cnpj'].notnull()]
data_with_geo.head()


# In[17]:

from geopy.distance import vincenty as distance
from IPython.display import display

x = data_with_geo.iloc[0]
display(x)
y = data_with_geo.iloc[20]
display(y)
distance(x[['latitude', 'longitude']],
         y[['latitude', 'longitude']])


# In[18]:

def coordinates_from_series(record):
    return record['latitude'], record['longitude']

coordinates = data_with_geo[['latitude', 'longitude']].     apply(coordinates_from_series, axis=1)
data_with_geo['coordinates'] = coordinates


# In[19]:

data_with_geo['coordinates'].head()


# In[20]:

is_in_brazil = (data_with_geo['longitude'] < -34.7916667) &     (data_with_geo['latitude'] < 5.2722222) &     (data_with_geo['latitude'] > -33.742222) &     (data_with_geo['longitude'] > -73.992222)
data_with_geo = data_with_geo[is_in_brazil]


# In[21]:

data_with_geo.head().iloc[0]


# In[22]:

from itertools import tee

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def distances(x):
    distance_list = [distance(*coordinates_pair).km
                     for coordinates_pair in pairwise(x)]
    return np.nansum(distance_list)

agg_data = data_with_geo.loc[data_with_geo['latitude'].notnull(),
                             ['congressperson_name', 'issue_date', 'coordinates', 'net_value']]
agg_data = agg_data.groupby(['congressperson_name', 'issue_date'])
results = agg_data.     agg({'coordinates': distances,
         'congressperson_name': len,
         'net_value': np.sum}). \
    rename(columns={'coordinates': 'distance_traveled',
                    'congressperson_name': 'count'})#. \
#     sort_values('distance_traveled', ascending=False). \
#     reset_index()


# In[23]:

results = results[results['distance_traveled'] > 0].     sort_values(['count', 'net_value'], ascending=False).     reset_index()


# In[24]:

results.head()


# In[25]:

sns.lmplot('count', 'distance_traveled',
           data=results,
           fit_reg=False,
           scatter_kws={"marker": "D", 
                        "s": 100})


# In[26]:

results[results['net_value'] > 800]['congressperson_name'].unique()


# In[27]:

predicate = (data_with_geo['congressperson_name'] == 'JOÃO CASTELO') &     (data_with_geo['issue_date'] == '2015-05-06')
records = data_with_geo[predicate]
len(records)


# In[28]:

x = records.iloc[0]
print(document_url(x))
x


# In[29]:

x = records.iloc[1]
print(document_url(x))
x


# In[30]:

x = records.iloc[2]
print(document_url(x))
x


# In[31]:

predicate = (data_with_geo['congressperson_name'] == 'MAJOR OLIMPIO') &     (data_with_geo['issue_date'] == '2015-11-09')
records = data_with_geo[predicate]
len(records)


# In[32]:

x = records.iloc[0]
print(document_url(x))
x


# In[33]:

x = records.iloc[1]
print(document_url(x))
x


# In[34]:

x = records.iloc[2]
print(document_url(x))
x


# In[35]:

d = read_csv()
d[d['document_id'] == '5847853'].iloc[0]


# In[36]:

predicate = (data_with_geo['congressperson_name'] == 'JOSÉ NUNES') &     (data_with_geo['issue_date'] == '2015-06-09')
records = data_with_geo[predicate]
len(records)


# In[37]:

x = records.iloc[0]
print(document_url(x))
x


# In[38]:

x = records.iloc[1]
print(document_url(x))
x


# In[39]:

predicate = (data_with_geo['congressperson_name'] == 'JOSÉ NUNES') &     (data_with_geo['issue_date'] == '2015-09-03')
records = data_with_geo[predicate]
len(records)


# In[40]:

x = records.iloc[0]
print(document_url(x))
x


# In[41]:

x = records.iloc[1]
print(document_url(x))
x


# In[42]:

predicate = (data_with_geo['congressperson_name'] == 'JOSÉ NUNES') &     (data_with_geo['issue_date'] == '2015-10-13')
records = data_with_geo[predicate]
len(records)


# In[43]:

x = records.iloc[0]
print(document_url(x))
x


# In[44]:

x = records.iloc[1]
print(document_url(x))
x


# In[45]:

aggregation[aggregation['len'] > 12]


# In[46]:

predicate = (data_with_geo['congressperson_name'] == 'CELSO MALDANER') &     (data_with_geo['issue_date'] == '2011-09-05')
records = data_with_geo[predicate]
len(records)


# In[47]:

records['net_value']


# In[48]:

records.apply(lambda row: print(('R$%i' % row['net_value']), document_url(row)),
              axis=1)


# In[49]:

records.iloc[0]


# In[50]:

records['document_id'].values


# In[56]:

data['congressperson_name'].unique()


# In[ ]:



