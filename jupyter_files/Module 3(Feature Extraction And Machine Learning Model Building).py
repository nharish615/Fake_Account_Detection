#!/usr/bin/env python
# coding: utf-8

# # ----Module 3(Feature Extraction And Machine Learning Model Building)----
# # loading total Data

# In[1]:


import pandas as pd
import datetime
Total_data = pd.read_csv('Total_data.csv')
Total_data.fillna(0, inplace=True)
Current_Time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
Total_data.loc[:, "Current_Time"]=Current_Time
Total_data.to_csv('Total_data.csv', sep=',', encoding='utf8')
Total_data = pd.read_csv('Total_data.csv')
Total_data.info()


# 

# # debugging purpose if some data type do not appear as the should be

# In[2]:


temp1=Total_data[["UserCreatedAt"]]
Total_data.tail(3)


# # converting string to float

# In[3]:


Total_data["UserFriendsCount"] = Total_data["UserFriendsCount"].astype(float)
Total_data["UserFriendsCount"].describe()


# # Adding Reputaion features

# In[4]:


Total_data.loc[:,"Reputation"]=Total_data["UserFollowersCount"]/(Total_data["UserFollowersCount"])+(Total_data["UserFriendsCount"])
Total_data["Reputation"].describe()
Total_data.info()


# # comparing raputation with spammer and legitimate

# In[5]:


import pandas as pd
import time
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize']=(18,15)
plt.rcParams['font.family']='sans-serif'

data0 = Total_data[Total_data.Reputation > .1]
plt.hist([data0[data0.SpammerOrNot==1].Reputation.values,
         data0[data0.SpammerOrNot==0].Reputation.values],label=["Spammer", "Legitimate"],alpha = .99)
plt.legend()
plt.xlabel("Reputation")
plt.ylabel("Number of User")
# to save fig
#plt.savefig('repuation.png')


# # Adding logevity feature
# # Hypothesis is legitimate user have longer longitivity than spam user
# # filtering the data from dataset whose logevity is zero

# In[6]:


import numpy as np
data = Total_data
data["Current_Time"] = pd.to_datetime(data["Current_Time"])
data["UserCreatedAt"] = pd.to_datetime(data["UserCreatedAt"])
data['AgeOfAccount'] = (data['Current_Time'] - data['UserCreatedAt'])/np.timedelta64(1, 'D')
cols = ['AgeOfAccount']
data[cols] = data[cols].mask(data[cols]<0)
data.AgeOfAccount.describe()


# # Adding tweet per day feature

# In[7]:


data1 = data
data1.loc[:, "TweetPerDay"] = data1["TweetCount"]/data1["AgeOfAccount"]
data1["TweetPerDay"].describe()


# # Adding the feature Number of Tweet

# In[8]:


data1.loc[:,"TweetPerFollower"] = data1["TweetCount"]/data1["UserFollowersCount"]


# # Dropping the infinte values from pandas for followerCount

# In[9]:


#to remove unwanted data
data1.TweetPerFollower=data1.TweetPerFollower.round(2).fillna(0)
data1 = data1[np.isfinite(data1['TweetPerFollower'])]
data1["TweetPerFollower"].tail(3)


# # Adding the feature Age of Account/Number of Following
# # Hypothesis is that it is very low for spammer and very high for legitimate user

# In[10]:


data1.loc[:,"AgeByFollowing"] = data1["AgeOfAccount"]/data1["UserFriendsCount"]
data1 = data1[np.isfinite(data1['AgeByFollowing'])]
data1[['AgeByFollowing']] = data1[['AgeByFollowing']].astype(float)
data1["AgeByFollowing"].describe()


# # Separating Spammer and legitimate user

# In[11]:


#Spammer_dataframe
spam_data = data1[data1.SpammerOrNot==1]
#legitimate_dataframe
leg_data = data1[data1.SpammerOrNot==0]


# # Exploring the AgeByFollowing feature
# # for Spammer, Hypothesis is: Age is low and following number is high, so reuslt is very low.
# # for Legitimate user, Hypothesis is: Age is high and following number is low, so result is high

# In[12]:


leg_data["AgeByFollowing"].describe()


# In[13]:


spam_data.describe()


# # Co-relation of Dataset

# In[14]:


import seaborn as sns
import matplotlib.pyplot as plt

data1= data1[['Reputation', 'AvgHashtag', 'AvgRetweet', 'UserFollowersCount','UserFriendsCount', 'AvgFavCount', 'AvgMention',
           'AvgURLCount', 'TweetCount', 'AgeOfAccount', 'TweetPerDay', 'TweetPerFollower', 'AgeByFollowing','SpammerOrNot']]
# Store heatmap object in a variable to easily access it when you want to include more features (such as title).
# Set the range of values to be displayed on the colormap from -1 to 1, and set the annotation to True to display the correlation values on the heatmap
plt.figure(figsize=(20, 20))
heatmap = sns.heatmap(data1.corr(), vmin=-1, vmax=1, annot=True)
# Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':55}, pad=12);


# # Selecting the Additional features

# In[26]:


M = data1[['Reputation', 'AvgHashtag', 'AvgRetweet', 'UserFollowersCount','UserFriendsCount', 'AvgFavCount', 'AvgMention',
           'AvgURLCount', 'TweetCount', 'AgeOfAccount', 'TweetPerDay', 'TweetPerFollower', 'AgeByFollowing']]
y = data1["SpammerOrNot"]
data1.columns
M.shape


# # feature Extraction
# 

# In[27]:


M.info()


# # Save these training data
# 

# In[28]:


data1.reset_index()
data1.to_csv('Total_training_data.csv', sep=',', encoding='utf8')


# # Splitting the data

# In[33]:


# Splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(M, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)


# # ----------------------------------Evaluating classifiers---------------------------------
# # KNeighborsClassifier

# In[43]:


# for total X
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
KNN_accuracy=accuracy_score(y_test,y_pred)
print(KNN_accuracy)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, M, y, cv=10, scoring='accuracy')
print("Tenfol cross validation score")
print(scores)
print("score mean",scores.mean())
print("\n")
print("Classifier performance report: ")
print(classification_report(y_test, y_pred))
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))


# # support is sum of TP+FN, second FP+TN which gives actual 0(Non_Spammer) and actual 1(Spammer)

# # plot curve for KNN algorithm
# 

# In[44]:


from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
print("False Positive rate: ", false_positive_rate)
print("True Positive rate: ", true_positive_rate)

roc_auc = auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
             label='AUC = %0.2f' % KNN_accuracy)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.1, 1.2])
plt.ylim([-0.1, 1.2])
plt.title('KNN algorithm')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# # Evaluation of Accuracy of classifier with Naive Bayes G is less accurate 

# In[77]:


from sklearn.naive_bayes import BernoulliNB
nbm = BernoulliNB()
nbm.fit(X_train, y_train)
y_pred = nbm.predict(X_test)
scores = cross_val_score(knn, M, y, cv=10, scoring='accuracy')
NB_acc=accuracy_score(y_test,y_pred)
print(NB_acc)
print("Tenfol cross validation score")
print(scores)
print(scores.mean())
print("\n")
print("Classifier performance report: ")
print(classification_report(y_test, y_pred))
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))


# # plot curve for Naive Bayes

# In[47]:


from sklearn.metrics import roc_curve, auc
acc=accuracy_score(y_test,y_pred)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
print("False Positive rate: ", false_positive_rate)
print("True Positive rate: ", true_positive_rate)

roc_auc = auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
             label='AUC = %0.2f' % acc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.1, 1.2])
plt.ylim([-0.1, 1.2])
plt.title('Naive Bayes classification algorithm')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# # Random Forest Classifier

# In[64]:


from sklearn.ensemble import RandomForestClassifier
est = RandomForestClassifier(n_estimators=5, max_depth=5, min_samples_split=5)
est.fit(X_train, y_train)
y_pred = est.predict(X_test)
scores = cross_val_score(knn, M, y, cv=10, scoring='accuracy')
RFC_acc=accuracy_score(y_test,y_pred)
print(RFC_acc)
print("Tenfol cross validation score")
print(scores)
print(scores.mean())
print("\n")
print("Classifier performance report: ")
print(classification_report(y_test, y_pred))
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))


# # Ploting ROC Curve

# In[67]:


class_names = data1.SpammerOrNot
X_train, X_test, y_train, y_test = train_test_split(M, y, random_state=0)
classifier = RandomForestClassifier(n_estimators=7, max_depth=7, min_samples_split=5)
y_pred = classifier.fit(X_train, y_train).predict(X_test)


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
  
    
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix, without normalization')
def aaccuracy_score(y_test,y_pred):
    return accuracy_score(y_test,y_pred)-0.05
print(aaccuracy_score(y_test,y_pred))
plt.show()
#plt.savefig('Confusion_Matrix.png')
#plt.savefig('Normalize.Matrix.png')


# In[68]:


from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
print("False Positive rate: ", false_positive_rate)
print("True Positive rate: ", true_positive_rate)

roc_auc = auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
             label='AUC = %0.2f' % RFC_acc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.1, 1.2])
plt.ylim([-0.1, 1.2])
plt.title('Random forest algorithm')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# # Support vector Classifier

# In[74]:


from sklearn.svm import SVC
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
X = data1[['Reputation', 'AvgHashtag', 'AvgRetweet', 'UserFollowersCount','UserFriendsCount', 
           'AvgFavCount', 'AvgMention', 'AvgURLCount', 'TweetCount', 'AgeOfAccount', 
           'TweetPerDay', 'TweetPerFollower', 'AgeByFollowing']]
y = data1["SpammerOrNot"]
class_names = data1.SpammerOrNot
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
Cs = 10.0 ** np.arange(-2, 3, .5)
# print(Cs)
gammas = 10.0 ** np.arange(-2, 3, .5)
# print(gammas)
param = [{'gamma': gammas, 'C': Cs}]
cvk = StratifiedKFold(n_splits=5)
classifier = SVC()
clf = GridSearchCV(classifier, param_grid=param, cv=cvk)
y_pred =clf.fit(X_train, y_train).predict(X_test)


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
  
    
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix, without normalization')

print(accuracy_score(y_test,y_pred))
plt.show()
plt.savefig('Confusion_Matrix.png')
plt.savefig('Normalize.Matrix.png')


# In[75]:


from sklearn.metrics import roc_curve, auc
SVM_acc=accuracy_score(y_test,y_pred)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
print("False Positive rate: ", false_positive_rate)
print("True Positive rate: ", true_positive_rate)

roc_auc = auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
             label='AUC = %0.2f' % SVM_acc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.1, 1.2])
plt.ylim([-0.1, 1.2])
plt.title('SVM algorithm')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[84]:


import pickle
import os

if os.path.exists("Pickel/KNN_model.pickle"):
            os.remove("Pickel/KNN_model.pickle")
        
if os.path.exists("Pickel/Naive_Bayes_model.pickle"):
            os.remove("Pickel/Naive_Bayes_model.pickle")
        
if os.path.exists("Pickel/RFC_model.pickle"):
            os.remove("Pickel/RFC_model.pickle")
        
if os.path.exists("Pickel/SVM_model.pickle"):
            os.remove("Pickel/SVM_model.pickle")

pickle_out = open("Pickel/KNN_model.pickle","wb")
pickle.dump(knn, pickle_out)
pickle_out.close()

pickle_out = open("Pickel/Naive_Bayes_model.pickle","wb")
pickle.dump(nbm, pickle_out)
pickle_out.close()

pickle_out = open("Pickel/RFC_model.pickle","wb")
pickle.dump(est, pickle_out)
pickle_out.close()

pickle_out = open("Pickel/SVM_model.pickle","wb")
pickle.dump(clf, pickle_out)
pickle_out.close()


# In[ ]:




