#!/usr/bin/env python
# coding: utf-8

# # -----------------Module 2(Data cleaning and Data Pre-processing)-------------------
# 
# # lodading legitimate User Data 

# In[1]:


import pandas as pd
Total_leg_data = pd.read_csv('Leg_data.csv')
Total_leg_data.fillna(0, inplace=True)
Total_leg_data.shape


# # drow bar plot to see tweet come from the which locations

# In[2]:


location_data = Total_leg_data['UserLocation'].value_counts()
location_data[2:15].plot(kind='bar', figsize=(14,7))


# # draw for a word how many times it used in tweets
# # Hypothesis is Legitimate users user very less compare to spammer

# In[4]:


import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (18,4)
plt.rcParams['font.family'] = 'sans-serif'
text = Total_leg_data['TextData']
is_sex = text.str.contains('sex')
is_sex=is_sex.astype(float)
is_sex.plot()


# # Save Followers count

# In[7]:


temp1 = Total_leg_data[["UserFollowersCount"]]
temp1.to_csv('C:/Users/asha/Desktop/Final Year Project Phase 2/Phase 2/Step2/userfollowerscount.csv', sep=',',encoding='utf8')


# # Retweet ratio also will be higher compare to spammer user

# In[8]:


Total_leg_data[['RetweetCount']] = Total_leg_data[['RetweetCount']].astype(float)
Total_leg_data.info()


# # to see how many people have zero tweet

# In[9]:


Total_leg_data = Total_leg_data[Total_leg_data.TweetCount!=0]
len(Total_leg_data[Total_leg_data.TweetCount<30])


# # adding New feature

# In[10]:


Total_leg_data.loc[:,"AvgHashtag"] = (Total_leg_data.groupby('UserID')["HashtagCount"].transform('sum'))/30
Total_leg_data.loc[:,"AvgURLCount"] = (Total_leg_data.groupby('UserID')["HttpCount"].transform('sum'))/30
Total_leg_data.loc[:,"AvgMention"] = (Total_leg_data.groupby('UserID')["MentionCount"].transform('sum'))/30
Total_leg_data.loc[:,"AvgRetweet"] = (Total_leg_data.groupby('UserID')["RetweetCount"].transform('sum'))/30
Total_leg_data.loc[:,"AvgFavCount"] = (Total_leg_data.groupby('UserID')["TweetFavouriteCount"].transform('sum'))/30


# # Selecting Repeted columns only and droping the repeted rows
# 

# In[13]:


unique_leg_row = Total_leg_data[["UserID", "UserScreenName", "UserCreatedAt", "UserDescriptionLength","UserFollowersCount", "UserFriendsCount", "UserLocation", "AvgHashtag", "AvgURLCount", "AvgMention", "AvgRetweet", "AvgFavCount", "TweetCount"]]
leg_data = unique_leg_row.drop_duplicates()
leg_data.info()


# #  Saving the reduced legitimate data

# In[16]:


fre = leg_data["UserFriendsCount"]
fre.to_csv("C:/Users/asha/Desktop/Final Year Project Phase 2/Phase 2/Step2/userfriendscount.csv", sep=',',encoding='utf8')


# # Datatype conversion from object to float
# 

# In[17]:


leg_data[['UserFriendsCount']] = leg_data[['UserFriendsCount']].astype(float)
leg_data.info()


# # Add a Column to Legitimate Data that this is not Spam =0

# In[25]:


leg_data.loc[:, "SpammerOrNot"]=0
leg_data.tail()


# In[26]:


leg_data["TweetCount"].describe()


# # Now Loading Spammer Data

# In[27]:


Total_spam_data = pd.read_csv("Spam_data.csv")
Total_spam_data.fillna(0, inplace=True)
Total_spam_data.shape


# # drow bar plot to see tweet come from the which locations

# In[22]:


get_ipython().run_line_magic('matplotlib', 'inline')
location_data = Total_spam_data['UserLocation'].value_counts()
location_data[2:15].plot(kind='bar', figsize=(14,7))


# # By Analyize Tweet I find that there is a lot of volgor word used by spam user compare to legitimate users

# In[28]:


import matplotlib.pyplot as plt
import string as str
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (18,4)
plt.rcParams['font.family'] = 'sans-serif'
text = Total_spam_data['TextData']
is_sex = text.str.contains('sex')
is_sex=is_sex.astype(float)
is_sex.plot()


# In[29]:


Total_spam_data=Total_spam_data.fillna(0)
Total_spam_data.shape


# In[30]:


temp2 = Total_spam_data[["UserFollowersCount"]]
temp2.to_csv('C:/Users/asha/Desktop/Final Year Project Phase 2/Phase 2/Step2/userfollowerscount1.csv', sep=',',encoding='utf8')


# # convert retweetcount object to float

# In[31]:


Total_spam_data[['RetweetCount']] = Total_spam_data[['RetweetCount']].astype(float)
Total_spam_data.info()


# # to see how many people have zero tweet

# In[32]:


Total_spam_data = Total_spam_data[Total_spam_data.TweetCount!=0]
len(Total_spam_data[Total_spam_data.TweetCount<30])


# # Adding new Extra feature 

# In[33]:


Total_spam_data.loc[:,'AvgHashtag'] = (Total_spam_data.groupby('UserID')["HashtagCount"].transform('sum'))/30
Total_spam_data.loc[:,'AvgURLCount'] = (Total_spam_data.groupby('UserID')["HttpCount"].transform('sum'))/30
Total_spam_data.loc[:,'AvgMention'] = (Total_spam_data.groupby('UserID')["MentionCount"].transform('sum'))/30
Total_spam_data.loc[:,'AvgRetweet'] = (Total_spam_data.groupby('UserID')["RetweetCount"].transform('sum'))/30
Total_spam_data.loc[:,'AvgFavCount'] = (Total_spam_data.groupby('UserID')["TweetFavouriteCount"].transform('sum'))/30


# In[34]:


Total_spam_data.tail(4)


# # Selecting Repeted columns only and droping the repeted rows

# In[35]:


unique_spam_row = Total_spam_data[["UserID", "UserScreenName", "UserCreatedAt", "UserDescriptionLength","UserFollowersCount", "UserFriendsCount", "UserLocation", "AvgHashtag", "AvgURLCount", "AvgMention", "AvgRetweet", "AvgFavCount", "TweetCount"]]
spam_data = unique_spam_row.drop_duplicates()
spam_data.info()


# # Saving the reduced Spammer data

# In[36]:


fre = spam_data["UserFriendsCount"]
fre.to_csv("C:/Users/asha/Desktop/Final Year Project Phase 2/Phase 2/Step2/userfriendscount1.csv", sep=',',encoding='utf8')


# #   userfriendscount Datatype conversion from object to float

# In[38]:


spam_data[['UserFriendsCount']] = spam_data[['UserFriendsCount']].astype(float)
spam_data.info()


# # Add a Column to Spammer Data that this is Spam =1

# In[42]:


spam_data.loc[:, "SpammerOrNot"]=1
spam_data.head()


# # Describe both legitimate user and spammer user of Twitter count

# In[43]:


spam_data["TweetCount"].describe()


# In[49]:


leg_data["TweetCount"].describe()


# # Merging the legitimate and spammer data

# In[48]:


frames = [leg_data, spam_data]
Total_data = pd.concat(frames, axis=0, sort=False)
Total_data.info()


# In[47]:


Total_data.reset_index()
Total_data.to_csv('Total_data.csv', sep=',', encoding='utf8')

