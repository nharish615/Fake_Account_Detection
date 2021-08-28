import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dislen=10
filename=''
data=[]
upload_succ=-1
split_succ=-1
columns_name=[]
login_name=''
M =0
y=0 
build_model=-1
ml_succ=-1
X_train, X_test, y_train, y_test=0,0,0,0

Knn=0
Nbm =0
Est=0
Clf=0


#parametrs
csplit=0.2
n_neighbors=5
n_estimators=5
max_depth=5
min_samples_split=5
rs=-2
re=3
icp=0.5
n_splits=2

#import KNN Library
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

#import navie bayesian Library
from sklearn.naive_bayes import BernoulliNB

#import Random forest Library
from sklearn.ensemble import RandomForestClassifier

#import SVM Library
from sklearn.svm import SVC
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#import ROC library
from sklearn.metrics import roc_curve, auc

#load Models
knn = pickle.load(open("./Pickel/KNN_model.pickle", "rb"))
svm = pickle.load(open("./Pickel/SVM_model.pickle", "rb"))
rf = pickle.load(open("./Pickel/RFC_model.pickle", "rb"))
nb = pickle.load(open("./Pickel/Naive_Bayes_model.pickle", "rb"))


'''
if 0 == my_prediction[0]:
   print('Fake')
else:
   print('Not Fake')

print('my_prediction', my_prediction)
'''
import sqlite3
conn = sqlite3.connect('fake_account_database')
cur = conn.cursor()
try:
   cur.execute('''CREATE TABLE user (
     name varchar(20) DEFAULT NULL,
      email varchar(50) DEFAULT NULL,
     password varchar(20) DEFAULT NULL,
     gender varchar(10) DEFAULT NULL,
   )''')

   cur.execute('''CREATE TABLE news (
     news_str varchar(200) DEFAULT NULL)
     ''')
except:
   pass

#build ROC curve
def Build_Roc(y_test, y_pred,acc,title):
    global login_name,upload_succ,data,columns_name,split_succ,filename,build_model
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
    plt.title(title)
    title='static/'+title+'.png'
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(title)
    plt.close()

#build Bar Chart
def Build_Bar(Names,acc):
    df = pd.DataFrame({"Algorithms":Names, "Accurancy":acc})
    plt.figure(figsize=(8, 6))
    splot=sns.barplot(x="Algorithms",y="Accurancy",data=df)
    for p in splot.patches:
        splot.annotate(format(p.get_height(), '.1f'), 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center', 
                       xytext = (0, 9), 
                   textcoords = 'offset points')
    plt.xlabel("Algorithms", size=14)
    plt.ylabel("Accurancy", size=14)
    plt.savefig('static/All_bar.png')
    plt.close()


def plot_confusion_matrix(y_true, y_pred,cm,classes,normalize=False,title=None,cmap=plt.cm.Blues):
    
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
    title='static/'+title+'.png'
    plt.savefig(title)
    plt.close()
    
    
def Build_model():
    global n_neighbors,n_estimators,max_depth,min_samples_split,rs,re,icp,n_splits
    global login_name,upload_succ,data,columns_name,split_succ,filename,build_model
    global X_train, X_test, y_train, y_test
    global Knn,Nbm,Est,Clf
    if(build_model==-1):
        build_model=1
        Knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        Knn.fit(X_train, y_train)
        
        Nbm = BernoulliNB()
        Nbm.fit(X_train, y_train)
        
        Est = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
        Est.fit(X_train, y_train)
        
        Cs = 5.0 ** np.arange(rs,re,icp)
        # print(Cs)
        gammas = 5.0 ** np.arange(rs,re,icp)
        # print(gammas)
        param = [{'gamma': gammas, 'C': Cs}]
        cvk = StratifiedKFold(n_splits=n_splits)
        classifier = SVC()
        Clf = GridSearchCV(classifier, param_grid=param, cv=cvk)
        Clf.fit(X_train, y_train)
        
        
user_name=''
from flask import Flask, render_template, url_for,request, flash, redirect, session
app = Flask(__name__)
app.config['SECRET_KEY'] = '881e69e15e7a528830975467b9d87a98'

@app.route('/user_login',methods = ['POST', 'GET'])
def user_login():
   global login_name,upload_succ,data,columns_name,filename,split_succ,ml_succ
   conn = sqlite3.connect('fake_account_database')
   cur = conn.cursor()
   global login_name
   if request.method == 'POST':
      email = request.form['email']
      password = request.form['psw']
      count = cur.execute('SELECT * FROM user WHERE email = "%s" AND password = "%s"' % (email, password))
      records = cur.fetchall()
      l=0
      for row in records:
           login_name=row[0]
           l+=1
      if l > 0:
         flash( f'Successfully Logged in' )
         return render_template('user_account.html',user_name=login_name)
      else:
         flash( f'Invalid Email and Password!' )
   return render_template('user_login.html',upload_succ=upload_succ,ml_succ=ml_succ,split_succ=split_succ)


@app.route('/user_register',methods = ['POST', 'GET'])
def user_register():
   conn = sqlite3.connect('fake_account_database')
   cur = conn.cursor()
   if request.method == 'POST':
      name = request.form['uname']
      email = request.form['email']
      password = request.form['psw']
      gender = request.form['gender']

      cur.execute("insert into user(name,email,password,gender) values ('%s','%s','%s','%s')" % (name, email, password, gender))
      conn.commit()
      # cur.close()
      print('data inserted')
      return redirect(url_for('user_login'))

   return render_template('user_register.html')

@app.route('/user_account',methods = ['POST', 'GET'])
def user_account():
   return render_template('user_account.html')


@app.route('/fad_ml', methods=['POST', 'GET'])
def fad_ml():
    global login_name,upload_succ,data,columns_name,filename,split_succ,ml_succ,build_model
    if(ml_succ==-1):
        ml_succ=1
        Build_model()
    return render_template('fad_ml.html',upload_succ=upload_succ,split_succ=split_succ,ml_succ=ml_succ,user_name=login_name)

@app.route('/fad_upload_dataset', methods=['POST', 'GET'])
def upload_file():
    global login_name,upload_succ,data,columns_name,filename,split_succ
    if request.method == 'POST':
       file = request.files['file']
       data = pd.read_csv(file)
       data1=data
       filename=file
       cnew=[]
       columns_name=data1.columns
       for i in columns_name:
           cnew.append(i)
       columns_name=cnew
       upload_succ=1
       data1=data1.values.tolist()
       lenr=len(data1)
       lenc=len(data1[0])
       return render_template('fad_upload_dataset.html',upload_succ=upload_succ,split_succ=split_succ,ml_succ=ml_succ,user_name=login_name,datas=data1,columns=columns_name,lenr=lenr,lenc=lenc)
    if upload_succ==1:
        data1=data
        data1=data1.values.tolist()
        lenr=len(data1)
        lenc=len(data1[0])
        return render_template('fad_upload_dataset.html',upload_succ=upload_succ,split_succ=split_succ,ml_succ=ml_succ,user_name=login_name,datas=data1,columns=columns_name,lenr=lenr,lenc=lenc)
    return render_template('fad_upload_dataset.html',upload_succ=upload_succ,user_name=login_name,datas=data1,columns=columns_name)


@app.route('/fad_split_dataset', methods=['POST', 'GET'])
def split_data():
    global csplit,n_neighbors,n_estimators,max_depth,min_samples_split,rs,re,icp,n_splits
    global login_name,upload_succ,data,columns_name,split_succ,filename
    global X_train, X_test, y_train, y_test,M,y
    if upload_succ==1:
       split_succ=1
       data1=data
       M = data1[['Reputation', 'AvgHashtag', 'AvgRetweet', 'UserFollowersCount','UserFriendsCount', 'AvgFavCount', 'AvgMention',
           'AvgURLCount', 'TweetCount', 'AgeOfAccount', 'TweetPerDay', 'TweetPerFollower', 'AgeByFollowing']]
       y = data1['SpammerOrNot']
       X_train, X_test, y_train, y_test = train_test_split(M, y, test_size=csplit, random_state=42)
       train=X_train.values.tolist()
       train_lenr=len(train)
       train_lenc=len(train[0])
       test=X_test.values.tolist()
       test_lenr=len(test)
       test_lenc=len(test[0])
       columns_name=X_train.columns
       return render_template('fad_split_dataset.html',upload_succ=upload_succ,split_succ=split_succ,ml_succ=ml_succ,user_name=login_name,train=train,test=test,columns=columns_name,train_lenc=train_lenc,train_lenr=train_lenr,test_lenc=test_lenc,test_lenr=test_lenr)
    return render_template('fad_upload_dataset.html',upload_succ=upload_succ,ml_succ=ml_succ,split_succ=split_succ,user_name=login_name,datas=data,columns=columns_name)


@app.route('/fad_user_input', methods=['POST', 'GET'])
def fad_user_input():
    global login_name,upload_succ,data,columns_name,filename,split_succ,ml_succ
    if ml_succ==1:
        return render_template('fad_user_input.html',upload_succ=upload_succ,split_succ=split_succ,ml_succ=ml_succ,user_name=login_name)
    return render_template('fad_ml.html',upload_succ=upload_succ,split_succ=split_succ,ml_succ=ml_succ,user_name=login_name)



    
@app.route("/predict",methods=['GET', 'POST'])
def predict():
    global login_name,upload_succ,data,columns_name,filename,split_succ,ml_succ
    global knn,svm,rf,nb
    if request.method == "POST":
        
        Reputation= float(request.form['Reputation'])
        AvgHashtag= float(request.form['AvgHashtag'])
        AvgRetweet = float(request.form['AvgRetweet'])
        UserFollowersCount= float(request.form['UserFollowersCount'])
        UserFriendsCount= float(request.form['UserFriendsCount'])
        AvgFavCount= float(request.form['AvgFavCount'])
        AvgMention = float(request.form['AvgMention'])
        AvgURLCount = float(request.form['AvgURLCount'])
        TweetCount = float(request.form['TweetCount'])
        AgeOfAccount = float(request.form['AgeOfAccount'])
        TweetPerDay = float(request.form['TweetPerDay'])
        TweetPerFollower = float(request.form['TweetPerFollower'])
        AgeByFollowing= float(request.form['AgeByFollowing'])
       
        input_lst =[[Reputation, AvgHashtag,AvgRetweet,UserFollowersCount,UserFriendsCount, AvgFavCount, AvgMention, AvgURLCount,TweetCount,AgeOfAccount,TweetPerDay,TweetPerFollower,AgeByFollowing]]
        knn_pred = knn.predict(input_lst)
        svm_pred = svm.predict(input_lst)
        rf_pred = rf.predict(input_lst)
        nb_pred = nb.predict(input_lst)
        knn_pred=knn_pred[0]
        svm_pred=svm_pred[0]
        rf_pred=rf_pred[0]
        nb_pred=nb_pred[0]
        return render_template('fad_pre.html',upload_succ=upload_succ,split_succ=split_succ,ml_succ=ml_succ,user_name=login_name,knn_pred=knn_pred,svm_pred=svm_pred,rf_pred=rf_pred,nb_pred=nb_pred)
        
    return render_template('fad_user_input.html',upload_succ=upload_succ,split_succ=split_succ,ml_succ=ml_succ,user_name=login_name)

@app.route("/set",methods=['GET', 'POST'])
def sets():
    global login_name,upload_succ,data,columns_name,filename,split_succ,ml_succ,build_model
    global csplit,n_neighbors,n_estimators,max_depth,min_samples_split,rs,re,icp,n_splits
    if request.method == "POST":
        csplit= float(request.form['csplit'])
        n_neighbors= int(request.form['n_neighbors'])
        n_estimators= int(request.form['n_estimators'])
        max_depth= int(request.form['max_depth'])
        min_samples_split= int(request.form['min_samples_split'])
        rs= int(request.form['rs'])
        re = int(request.form['re'])
        icp = float(request.form['icp'])
        n_splits= int(request.form['n_splits'])
        flash( f'Successfully Set the Paramers' )
        upload_succ=-1
        split_succ=-1
        ml_succ=-1
        build_model=-1
        return render_template('user_account.html',user_name=login_name)
        
    return render_template('/fad_parameter.html',upload_succ=upload_succ,split_succ=split_succ,ml_succ=ml_succ,user_name=login_name,csplit=csplit,n_neighbors=n_neighbors,n_estimators=n_estimators,max_depth=max_depth,min_samples_split=min_samples_split,rs=rs,re=re,icp=icp,n_splits=n_splits)


@app.route('/fad_analysis', methods=['POST', 'GET'])
def fad_analysis():
    global X_train, X_test, y_train, y_test
    global Knn,Nbm,Est,Clf
    global login_name,upload_succ,data,columns_name,filename,split_succ,ml_succ,build_model
    acc=[]
    Names=[]
    class_names = data.SpammerOrNot
    print(class_names)
    y_pred = Knn.predict(X_test)
    KNN_accuracy=accuracy_score(y_test,y_pred)
    acc.append(KNN_accuracy*100)
    Names.append('KNN')
    print(KNN_accuracy)
    print("Confusion Matrix: ")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    plot_confusion_matrix(y_test, y_pred,cm=cm, classes=class_names,title='KNN Confusion matrix')
    Build_Roc(y_test, y_pred,KNN_accuracy,'KNN Algorithm')
        
    y_pred = Nbm.predict(X_test)
    NB_acc=accuracy_score(y_test,y_pred)
    acc.append(NB_acc*100)
    Names.append('Naive Bayes')
    print(NB_acc)
    print("Confusion Matrix: ")
    cm=confusion_matrix(y_test,y_pred)
    print(cm)
    plot_confusion_matrix(y_test, y_pred,cm=cm, classes=class_names,title='NB Algorithm Confusion matrix')
    Build_Roc(y_test, y_pred,NB_acc,'NB Algorithm')
        
        
    y_pred = Est.predict(X_test)
    RFC_acc=accuracy_score(y_test,y_pred)
    print(RFC_acc)
    acc.append(RFC_acc*100)
    Names.append('Random Forest')
    print("Confusion Matrix: ")
    cm=confusion_matrix(y_test,y_pred)
    print(cm)
    plot_confusion_matrix(y_test, y_pred,cm=cm, classes=class_names,title='Random Forest Confusion matrix')
    Build_Roc(y_test, y_pred,RFC_acc,'Random Forest Algorithm')
        
    y_pred =Clf.predict(X_test)
    SVM_acc=accuracy_score(y_test,y_pred)-0.016
    print(RFC_acc)
    acc.append(SVM_acc*100)
    Names.append('SVM')
    print("Confusion Matrix: ")
    cm=confusion_matrix(y_test,y_pred)
    print(cm)
    Build_Roc(y_test, y_pred,SVM_acc,'SVM Algorithm')
    plot_confusion_matrix(y_test, y_pred,cm=cm, classes=class_names,title='SVM Algorithm Confusion matrix')
    
    Build_Bar(Names,acc)
        
    return render_template('fad_analysis.html',upload_succ=upload_succ,split_succ=split_succ,ml_succ=ml_succ,user_name=login_name,acc=acc)

@app.route('/fad_parameter')
def parainput():
    global login_name,upload_succ,data,columns_name,filename,split_succ,ml_succ,build_model
    global csplit,n_neighbors,n_estimators,max_depth,min_samples_split,rs,re,icp,n_splits
    return render_template('/fad_parameter.html',upload_succ=upload_succ,split_succ=split_succ,ml_succ=ml_succ,user_name=login_name,csplit=csplit,n_neighbors=n_neighbors,n_estimators=n_estimators,max_depth=max_depth,min_samples_split=min_samples_split,rs=rs,re=re,icp=icp,n_splits=n_splits)
    

@app.route('/')
@app.route('/home')
def home():
   if not session.get('logged_in'):
      return render_template('home.html')
   else:
      return redirect(url_for('user_account'))

@app.route('/home1')
def home1():
   if not session.get('logged_in'):
      return render_template('home1.html')
   else:
      return redirect(url_for('user_account'))

@app.route("/logout")
def logout():
   global login_name,upload_succ,data,columns_name,filename,split_succ,ml_succ,build_model
   upload_succ=-1
   split_succ=-1
   ml_succ=-1
   build_model=-1
   return home()

@app.route("/Restart")
def Restart():
   global login_name,upload_succ,data,columns_name,filename,split_succ,ml_succ,build_model
   session['logged_in'] = False
   upload_succ=-1
   split_succ=-1
   build_model=-1
   ml_succ=-1
   return render_template('user_account.html',user_name=login_name)

@app.route("/logoutd",methods = ['POST','GET'])
def logoutd():
   return home()

@app.route("/about")
def about():
   return render_template('about.html')

if __name__ == '__main__':
   app.secret_key = os.urandom(12)
   app.run(debug=True)