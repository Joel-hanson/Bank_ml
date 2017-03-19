
# coding: utf-8

# In[380]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pandas as pd


# In[381]:

dataframe = pd.read_csv("train_indessa.csv")
testframe = pd.read_csv("train_indessa.csv")
dataframe =dataframe.drop(['member_id','emp_title','zip_code','addr_state','dti','batch_enrolled','sub_grade','desc','title','verification_status_joint'],axis=1)

testframe =testframe.drop(['member_id','emp_title','zip_code','addr_state','dti','batch_enrolled','sub_grade','desc','title','verification_status_joint'],axis=1)



# In[382]:

# dataframe.loc[:,('nstatus')] = dataframe['loan_status'] == 0
# dataframe.loc[:,('nstatus')] = dataframe['nstatus'].astype(int)
dataframe.loc[:,('term')] = dataframe['term'].str.split('months').str[0]
dataframe.loc[dataframe.grade == 'A'  ,('grade')] = '1' 
dataframe.loc[dataframe.grade == 'B'  ,('grade')] = '2' 
dataframe.loc[dataframe.grade == 'C'  ,('grade')] = '3' 
dataframe.loc[dataframe.grade == 'D'  ,('grade')] = '4' 
dataframe.loc[dataframe.grade == 'E'  ,('grade')] = '5' 
dataframe.loc[:,('emp_length')] = dataframe['emp_length'].str.split('years').str[0]
dataframe.loc[:,('emp_length')] = dataframe['emp_length'].str.split('year').str[0]
dataframe.loc[dataframe.emp_length == "< 1 "  ,('emp_length')] = '0' 
dataframe.loc[dataframe.emp_length == "10+ "  ,('emp_length')] = '11' 
dataframe.loc[dataframe.home_ownership == "OWN"  ,('home_ownership')] = '1' 
dataframe.loc[dataframe.home_ownership == "RENT"  ,('home_ownership')] = '2' 
dataframe.loc[dataframe.home_ownership == "MORTGAGE"  ,('home_ownership')] = '3'
dataframe.loc[dataframe.verification_status == "Source Verified"  ,('verification_status')] = '1'
dataframe.loc[dataframe.verification_status == "Verified"  ,('verification_status')] = '2'
dataframe.loc[dataframe.verification_status == "Not Verified"  ,('verification_status')] = '3'
dataframe.loc[dataframe.mths_since_last_major_derog.isnull()  ,('mths_since_last_major_derog')] = '0'
dataframe.loc[dataframe.pymnt_plan == 'n'  ,('pymnt_plan')] = '1'
dataframe.loc[dataframe.purpose == 'debt_consolidation'  ,('purpose')] = '3'
dataframe.loc[dataframe.purpose == 'home_improvement'  ,('purpose')] = '2'
dataframe.loc[dataframe.purpose == 'credit_card'  ,('purpose')] = '3'
dataframe.loc[dataframe.purpose == 'major_purchase'  ,('purpose')] = '4'
dataframe.loc[dataframe.purpose == 'other'  ,('purpose')] = '5'

dataframe.loc[dataframe.mths_since_last_delinq.isnull()  ,('mths_since_last_delinq')] = '0'
dataframe.loc[dataframe.mths_since_last_record.isnull()  ,('mths_since_last_record')] = '0'
dataframe.loc[dataframe.initial_list_status == 'f'  ,('initial_list_status')] = '1'
dataframe.loc[dataframe.initial_list_status == 'w'  ,('initial_list_status')] = '0'
dataframe.loc[dataframe.application_type == 'INDIVIDUAL'  ,('application_type')] = '1'
dataframe.loc[dataframe.application_type == 'JOINT'  ,('application_type')] = '0'
dataframe.loc[:,('last_week_pay')] = dataframe['last_week_pay'].str.split('th week').str[0]


testframe.loc[:,('term')] = testframe['term'].str.split('months').str[0]
testframe.loc[testframe.grade == 'A'  ,('grade')] = '1' 
testframe.loc[testframe.grade == 'B'  ,('grade')] = '2' 
testframe.loc[testframe.grade == 'C'  ,('grade')] = '3' 
testframe.loc[testframe.grade == 'D'  ,('grade')] = '4' 
testframe.loc[testframe.grade == 'E'  ,('grade')] = '5' 
testframe.loc[testframe.grade == 'F'  ,('grade')] = '6' 
testframe.loc[:,('emp_length')] = testframe['emp_length'].str.split('years').str[0]
testframe.loc[:,('emp_length')] = testframe['emp_length'].str.split('year').str[0]
testframe.loc[testframe.emp_length == "< 1 "  ,('emp_length')] = '0' 
testframe.loc[testframe.emp_length == "10+ "  ,('emp_length')] = '11' 
testframe.loc[testframe.home_ownership == "OWN"  ,('home_ownership')] = '1' 
testframe.loc[testframe.home_ownership == "RENT"  ,('home_ownership')] = '2' 
testframe.loc[testframe.home_ownership == "MORTGAGE"  ,('home_ownership')] = '3'
testframe.loc[testframe.verification_status == "Source Verified"  ,('verification_status')] = '1'
testframe.loc[testframe.verification_status == "Verified"  ,('verification_status')] = '2'
testframe.loc[testframe.verification_status == "Not Verified"  ,('verification_status')] = '3'
testframe.loc[testframe.mths_since_last_major_derog.isnull()  ,('mths_since_last_major_derog')] = '0'
testframe.loc[testframe.pymnt_plan == 'n'  ,('pymnt_plan')] = '1'
testframe.loc[testframe.purpose == 'debt_consolidation'  ,('purpose')] = '3'
testframe.loc[testframe.purpose == 'home_improvement'  ,('purpose')] = '2'
testframe.loc[testframe.purpose == 'credit_card'  ,('purpose')] = '3'
testframe.loc[testframe.mths_since_last_delinq.isnull()  ,('mths_since_last_delinq')] = '0'
testframe.loc[testframe.mths_since_last_record.isnull()  ,('mths_since_last_record')] = '0'
testframe.loc[testframe.initial_list_status == 'f'  ,('initial_list_status')] = '1'
testframe.loc[testframe.initial_list_status == 'w'  ,('initial_list_status')] = '0'
testframe.loc[testframe.application_type == 'INDIVIDUAL'  ,('application_type')] = '1'
testframe.loc[testframe.application_type == 'JOINT'  ,('application_type')] = '0'
testframe.loc[:,('last_week_pay')] = testframe['last_week_pay'].str.split('th week').str[0]
testframe.loc[testframe.tot_coll_amt.isnull()  ,('tot_coll_amt')] = '0'
testframe.loc[testframe.tot_cur_bal.isnull()  ,('tot_cur_bal')] = '0'
testframe.loc[testframe.total_rev_hi_lim.isnull()  ,('total_rev_hi_lim')] = '0'
testframe.loc[testframe.purpose == 'major_purchase'  ,('purpose')] = '4'
testframe.loc[testframe.last_week_pay == 'NA'  ,('last_week_pay')] = '0'
testframe.loc[testframe.purpose == 'other'  ,('purpose')] = '5'

# testframe


# In[383]:

dataframe.loc[:]=dataframe[:].astype('float32')
# dataframe


# In[384]:

testframe.loc[:] = testframe[:].astype('float32')
# testframe


# In[385]:

inputX = dataframe.ix[:,:34]
inputY = dataframe.loc[:,['loan_status']]
inputX_ = testframe.ix[:,:34]
inputY_ = testframe.loc[:,['loan_status']]

# inputX = np.array(inputX, dtype='float32')
# inputX


# In[386]:

# inputX


# In[ ]:




# In[387]:

# LOAN_TRAINING = dataframe
# LOAN_TEST = inputY
# inputY_


# In[388]:

# training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
#     filename=LOAN_TRAINING,
#     target_dtype=np.int,
#     features_dtype=np.float32)
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=34)]


# In[389]:

classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=2,
                                            model_dir="/tmp/loan2_model")
# feature_columns


# In[390]:

classifier.fit(x=inputX, y=inputY, steps=2000)


# In[391]:

accuracy_score = classifier.evaluate(x=inputX_, y=inputY)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))


# In[392]:

# inputX_.loc[[23]]


# In[393]:


y = list(classifier.predict(inputX_.loc[[23]], as_iterable=True))
print('Predictions: {}'.format(str(y)))


# In[ ]:



