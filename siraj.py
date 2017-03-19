
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import tensorflow as tf


# In[2]:

dataframe = pd.read_csv("train_indessa.csv")
dataframe =dataframe.drop(['member_id','emp_title','zip_code','addr_state','dti','batch_enrolled','sub_grade','desc','title','verification_status_joint'],axis=1)
dataframe = dataframe[0:10]


# In[3]:

dataframe


# In[4]:

dataframe.loc[:,('nstatus')] = dataframe['loan_status'] == 0
dataframe.loc[:,('nstatus')] = dataframe['nstatus'].astype(int)
dataframe.loc[:,('term')] = dataframe['term'].str.split('months').str[0]
dataframe.loc[dataframe.grade == 'A'  ,('grade')] = '1' 
dataframe.loc[dataframe.grade == 'B'  ,('grade')] = '2' 
dataframe.loc[dataframe.grade == 'C'  ,('grade')] = '3' 
dataframe.loc[dataframe.grade == 'D'  ,('grade')] = '4' 
dataframe.loc[dataframe.grade == 'E'  ,('grade')] = '5' 
dataframe.loc[:,('emp_length')] = dataframe['emp_length'].str.split('years').str[0]
dataframe.loc[dataframe.emp_length == "< 1 year"  ,('emp_length')] = '0' 
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
dataframe.loc[dataframe.mths_since_last_delinq.isnull()  ,('mths_since_last_delinq')] = '0'
dataframe.loc[dataframe.mths_since_last_record.isnull()  ,('mths_since_last_record')] = '0'
dataframe.loc[dataframe.initial_list_status == 'f'  ,('initial_list_status')] = '1'
dataframe.loc[dataframe.initial_list_status == 'w'  ,('initial_list_status')] = '0'
dataframe.loc[dataframe.application_type == 'INDIVIDUAL'  ,('application_type')] = '1'
dataframe.loc[dataframe.application_type == 'JOINT'  ,('application_type')] = '0'
dataframe.loc[:,('last_week_pay')] = dataframe['last_week_pay'].str.split('th week').str[0]


dataframe


# In[5]:

inputX = dataframe.ix[:,:34].as_matrix()
inputY = dataframe.loc[:,['loan_status','nstatus']].as_matrix()

inputX


# In[6]:

inputY


# In[7]:

learning_rate = 0.000001
training_epochs = 4000
display_steps = 50
n_sample = inputY.size


# In[8]:

x = tf.placeholder(tf.float32,[None,34])
W = tf.Variable(tf.zeros([34,2]))
b = tf.Variable(tf.zeros([2]))
y_values = tf.add(tf.matmul(x,W),b)
y = tf.nn.softmax(y_values)
y_ = tf.placeholder(tf.float32,[None,2])
cost = tf.reduce_sum(tf.pow(y_ - y, 2)) / (2*n_sample)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# In[9]:

sess = tf.Session()
sess.run(tf.initialize_all_variables())


# In[10]:

for i in range(training_epochs):
    sess.run(optimizer,feed_dict = {x:inputX,y_:inputY})
    
    if (i) % display_steps == 0:
        cc = sess.run(cost, feed_dict = {x:inputX,y_:inputY})
        print "training step:", '%04d' % (i), "cost=","{:.9f}".format(cc)
        
print "optimization finished!"
training_cost = sess.run(cost,feed_dict={x: inputX,y_:inputY})
print "training cost=",training_cost, "W=", sess.run(W), "b=", sess.run(b),'\n'


# In[11]:

sess.run(y, feed_dict = {x:inputX})


# In[ ]:




# In[ ]:



