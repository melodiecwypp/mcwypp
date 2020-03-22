#!/usr/bin/env python
# coding: utf-8

# In[188]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import metrics


# In[189]:


df=pd.read_csv('train.csv')


# In[190]:


df.head()


# In[191]:


df.describe()


# In[192]:


df.info()


# In[193]:


df=df.drop(['id','purchaseTime'],axis=1)


# In[194]:


df.head()


# In[195]:


df=df.drop(['purchaseTime'],axis=1)


# In[196]:


plt.plot(df['label'])


# In[197]:


def recode(series):
    if series == -1:
        return 0
    else:
        return series
df['label'] = df['label'].apply(recode)


# In[198]:


df['label'].value_counts()


# In[199]:


xdata=df.drop(['label'],1)
y=df['label']


# In[200]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(xdata, y, test_size=0.2, random_state=21, stratify=y)


# In[201]:


y_test


# In[202]:


get_ipython().system('pip install imblearn')
get_ipython().system('pip install Tensorflow')
get_ipython().system('pip install imbalanced-learn --user')
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(sampling_strategy=0.01)
X_rus, y_rus = rus.fit_sample(x_train, y_train)


# In[ ]:





# In[203]:


y.value_counts()


# In[204]:


y_rus.value_counts()


# In[163]:


plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), annot = True)


# In[ ]:





# In[205]:


#logistic regression

import statsmodels.api as sm
mlog = sm.Logit(y, xdata)
result = mlog.fit()
print(result.summary())


# In[215]:


X_rus_lr = X_rus[['visitTime', 'C6','C7', 'C9', 'N3', 'N4', 'N5', 'N6', 'N8', 'N9']]
X_test_lr = x_test[['visitTime', 'C6','C7', 'C9', 'N3', 'N4', 'N5', 'N6', 'N8', 'N9']]


# In[216]:


from sklearn.linear_model import LogisticRegression
mlr = LogisticRegression(random_state=42, class_weight='balanced')
result = mlr.fit(X_rus_lr, y_rus)


# In[217]:


mlrpred = mlr.predict(X_test_lr)
mlrprob = mlr.predict_proba(X_test_lr)


# In[218]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,mlrpred)


# In[219]:


from sklearn.metrics import confusion_matrix, classification_report
lrcm = confusion_matrix(y_test,mlrpred)
lrcm


# In[220]:


roc = metrics.roc_auc_score(y_test, mlrpred)
roc


# In[274]:


test=pd.read_csv('test1.csv')


# In[275]:


test


# In[269]:


test=test.drop(['id','purchaseTime'],axis=1)


# In[276]:


def recode(series):
    if series == -1:
        return 0
    else:
        return series
test['label'] = test['label'].apply(recode)


# In[277]:


xdatatest=test.drop(['label'],1)
y=test['label']


# In[281]:


xdatatest


# In[278]:


xxdatatest = xdatatest[['visitTime', 'C6','C7', 'C9', 'N3', 'N4', 'N5', 'N6', 'N8', 'N9']]


# In[279]:


mlrpredtest = mlr.predict(xxdatatest)
mlrprobtest = mlr.predict_proba(xxdatatest)


# In[282]:


predictionstest = pd.Series(data=mlrpredtest,index=xdatatest.id, name='predicted_value')
predictionstest


# In[284]:


proba = pd.DataFrame(data=mlrprobtest, index=xdatatest.id)
proba = proba.drop([0], 1)
proba.rename(columns={1: 'Probs'})


# In[285]:


probatable=proba.to_csv('probatable.csv',index=True)


# In[286]:


probatable


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




