#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# In[2]:


loan_data_org = pd.read_csv("E:/Egerton/Lending club data2.csv")


# In[3]:


loan_data = loan_data_org.copy()


# In[4]:


loan_data


# In[5]:


pd.options.display.max_columns = None


# In[6]:


loan_data


# In[7]:


loan_data.columns.values


# In[8]:


loan_data.info()


# ## Data preprocessing

# In[9]:


loan_data["emp_length"].unique()


# In[14]:


loan_data["emp_length_int"] = loan_data["emp_length"].str.replace('\+ years', "")
loan_data["emp_length_int"] = loan_data["emp_length_int"].str.replace("< 1 year", str(0))
loan_data["emp_length_int"] = loan_data["emp_length_int"].str.replace("nan", str(0))
loan_data["emp_length_int"] = loan_data["emp_length_int"].str.replace("years", "")
loan_data["emp_length_int"] = loan_data["emp_length_int"].str.replace("year", "")


# In[15]:


loan_data["emp_length_int"].unique()


# In[16]:


type(loan_data["emp_length_int"][0])


# In[17]:


loan_data["emp_length_int"] = pd.to_numeric(loan_data["emp_length_int"])


# In[18]:


type(loan_data["emp_length_int"][0])


# In[19]:


loan_data["term"].unique()


# In[20]:


loan_data["term_int"] = loan_data["term"].str.replace('months', "")
loan_data["term_int"] = loan_data["term_int"].str.replace('n/a', str(0))


# In[21]:


type(loan_data["term_int"][0])


# In[22]:


loan_data["term_int"] = pd.to_numeric(loan_data["term_int"])


# In[23]:


type(loan_data["term_int"][0])


# String variables to date variables

# In[24]:


loan_data["earliest_cr_line"]


# In[25]:


loan_data["earliest_cr_line_date"] = pd.to_datetime(loan_data["earliest_cr_line"], format = "%Y-%M")


# In[26]:


type(loan_data["earliest_cr_line_date"][0])


# In[27]:


loan_data["earliest_cr_line_date"]


# In[28]:


pd.to_datetime("2017-12-01") -loan_data["earliest_cr_line_date"]


# In[29]:


loan_data["mths_earliest_cr_line_date"] = round(pd.to_numeric((pd.to_datetime("2017-12-01") - loan_data["earliest_cr_line_date"]) / np.timedelta64(1, "M")))


# In[30]:


loan_data["mths_earliest_cr_line_date"].describe()


# In[31]:


loan_data.info()


# In[32]:


pd.get_dummies(loan_data["grade"])


# In[33]:


pd.get_dummies(loan_data["grade"], prefix = "grade", prefix_sep= ":")


# In[34]:


loan_data_dummies = [pd.get_dummies(loan_data["grade"], prefix = "grade", prefix_sep= ":"),
                    pd.get_dummies(loan_data["sub_grade"], prefix = "sub_grade", prefix_sep= ":"),
                    pd.get_dummies(loan_data["home_ownership"], prefix = "home_owership", prefix_sep= ":"),
                    pd.get_dummies(loan_data["verification_status"], prefix = "verification_status", prefix_sep= ":"),
                    pd.get_dummies(loan_data["loan_status"], prefix = "loan_status", prefix_sep= ":"),
                    pd.get_dummies(loan_data["purpose"], prefix = "purpose", prefix_sep= ":"),
                    pd.get_dummies(loan_data["addr_state"], prefix = "addr_state", prefix_sep= ":"),
                    pd.get_dummies(loan_data["initial_list_status"], prefix = "initial_list_status", prefix_sep= ":")]


# In[35]:


loan_data_dummies = pd.concat(loan_data_dummies, axis=1)


# In[36]:


type(loan_data_dummies)


# In[37]:


loan_data = pd.concat([loan_data, loan_data_dummies], axis=1)


# In[38]:


loan_data.head(2)


# In[39]:


#loan_data.columns.values


# ## Checking the values and cleaning

# In[40]:


loan_data.isnull()


# In[41]:


pd.options.display.max_rows = None
loan_data.isnull().sum()


# In[42]:


# 'Total revolving high credit/ credit limit', so it makes sense that the missing values are equal to funded_amnt.
loan_data['total_rev_hi_lim'].fillna(loan_data['funded_amnt'], inplace=True)
# We fill the missing values with the values of another variable.


# In[43]:


loan_data['total_rev_hi_lim'].isnull().sum()


# In[44]:


loan_data['annual_inc'].fillna(loan_data['annual_inc'].mean(), inplace=True)
# We fill the missing values with the mean value of the non-missing values.


# In[45]:


#loan_data['mths_since_earliest_cr_line'].fillna(0, inplace=True)
loan_data['acc_now_delinq'].fillna(0, inplace=True)
loan_data['total_acc'].fillna(0, inplace=True)
loan_data['pub_rec'].fillna(0, inplace=True)
loan_data['open_acc'].fillna(0, inplace=True)
loan_data['inq_last_6mths'].fillna(0, inplace=True)
loan_data['delinq_2yrs'].fillna(0, inplace=True)
loan_data['emp_length_int'].fillna(0, inplace=True)
# We fill the missing values with zeroes.


# In[46]:


#pd.options.display.max_rows = None
loan_data.isnull().sum()


# ## PD Model

# ### Data Preparation

# #### Dependant variable (Good /Bad)

# In[47]:


loan_data["loan_status"].unique()


# In[48]:


loan_data["loan_status"].value_counts()


# In[49]:


loan_data["loan_status"].value_counts()/loan_data["loan_status"].count()


# In[50]:


loan_data["good_bad"] = np.where(loan_data["loan_status"].isin(["Charged Off", "Does not meet the credit policy. Status:Charged Off", "Late (31-120 days)", "Default"]),0,1)


# In[51]:


loan_data["good_bad"]


# In[53]:


#train_test_split(loan_data.drop("good_bad", axis=1),loan_data["good_bad"])


# In[ ]:




