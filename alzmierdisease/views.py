from django.shortcuts import render
from django.http import HttpResponse
from django.contrib import messages
from docutils.nodes import inline

from patient.models import *
from patient.forms import *
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from django_pandas.io import read_frame
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import datasets

#svm packages
import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.preprocessing import MinMaxScaler


from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_validate, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
classification_report
from sklearn.neighbors import KNeighborsClassifier

from helpers.my_one_hot_encoder import MyOneHotEncoder



def index(request):
    return render(request,'index.html')

def activatepatient(request):
    if request.method == 'GET':
        uname = request.GET.get('pid')
        print(uname)
        status = 'Activated'
        print("pid=", uname, "status=", status)
        patientModel.objects.filter(id=uname).update(status=status)
        qs = patientModel.objects.all()
        return render(request, 'admin/patientdetails.html',{"object": qs})


def base(request):
    return render(request, "base.html")


def adminlogin(request):
    return render(request,"admin/adminlogin.html")

def adminloginaction(request):
    if request.method == "POST":
        if request.method == "POST":
            login = request.POST.get('username')
            print(login)
            pswd = request.POST.get('password')
            if login == 'admin' and pswd == 'admin':
                return render(request,'admin/adminhome.html')
            else:
                messages.success(request, 'Invalid user id and password')
    #messages.success(request, 'Invalid user id and password')
    return render(request,'admin/adminlogin.html')

def logout(request):
    return render(request,'index.html')


def svm(request):
    # qs = storedatamodel.objects.all()
    # data = read_frame()
    data=pd.read_csv(r'C:\Users\Hi\Desktop\alzheimer project\3 Alzheimer Disease Prediction using Machine Learning Algorithms\Code\alzmierdisease\data_small.csv')
    data = data.fillna(data.mean())
    # data[0:label]
    # data.info()
    print(data.head())
    # print(data.describe())
    #print("data-label:",data.label)
    dataset = data.iloc[:,[6,7]].values
    print("x",dataset)
    dataset1 = data.iloc[:,-1].values
    print("y",dataset1)
    print("shape",dataset.shape)
    X = dataset
    y = dataset1
    print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3,random_state=0)
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)
    #print(svclassifier.predict([0.58, 0.76]))
    y_pred = svclassifier.predict(X_test)
    m = confusion_matrix(y_test, y_pred)
    accurancy = classification_report(y_test, y_pred)
    print(m)
    print(accurancy)
    x = accurancy.split()
    print("Toctal splits ", len(x))
    dict = {

        "m": m,
        "accurancy": accurancy,
        'len0': x[0],
        'len1': x[1],
        'len2': x[2],
        'len3': x[3],
        'len4': x[4],
        'len5': x[5],
        'len6': x[6],
        'len7': x[7],
        'len8': x[8],
        'len9': x[9],
        'len10': x[10],
        'len11': x[11],
        'len12': x[12],
        'len13': x[13],
        'len14': x[14],
        'len15': x[15],
        'len16': x[16],
        'len17': x[17],
        'len18': x[18],
        'len19': x[19],
        'len20': x[20],
        'len21': x[21],
        'len22': x[22],
        'len23': x[23],
        'len24': x[24],
        'len25': x[25],
        'len26': x[26],
        'len27': x[27],
        'len28': x[28],
        # 'len29': x[29],
        # 'len30': x[30],
        # 'len31': x[31],
        # 'len32': x[32],
        # 'len33': x[33],

    }
    return render(request, 'admin/accuracy.html', dict)




def testing(request):
    data = pd.read_csv('data_small.csv')
    headers = data.columns.values
    # print(headers)
    print(data.tail())
    data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], inplace=True)
    data.tail()
    data['VIS'] = [0 if vis == "bl" else vis[1:] for vis in data['VISCODE']]
    data['VIS'] = pd.to_numeric(data['VIS'])
    print(data.VIS.head())
    datap1 = data.query('VISCODE == "bl" and (DXCHANGE == 1 or DXCHANGE == 3)')
    datap1.drop(columns=['VISCODE'], inplace=True)
    datap1.DXCHANGE.value_counts()

    null_columns = datap1.columns[datap1.isnull().any()]
    print(datap1[null_columns].isnull().sum())

    interested_columns = ['DXCHANGE', 'RID', 'PTID', 'EXAMDATE', 'AGE', 'PTGENDER', 'PTEDUCAT', 'PTMARRY', 'APOE4','CDRSB', 'ADAS11', 'ADAS13', 'MMSE', 'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_forgetting','RAVLT_perc_forgetting', 'FAQ', 'MOCA', 'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp','ICV', 'FDG', 'PIB',  'AV45']
    print(datap1[interested_columns].isnull().sum())
    datap1pred = datap1[['PTGENDER', 'PTMARRY', 'AGE', 'PTEDUCAT', 'APOE4', 'CDRSB', 'ADAS11', 'ADAS13', 'MMSE', \
                         'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_forgetting', 'RAVLT_perc_forgetting', 'FAQ', \
                         'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp', 'ICV',
                         'DXCHANGE']]
    print(datap1pred.reset_index(inplace=True, drop=True))
    num_imp = IterativeImputer(max_iter=20).fit_transform(datap1pred.select_dtypes(exclude=[object]))
    datap1predi = pd.concat([datap1pred.select_dtypes(include=[object]), pd.DataFrame(num_imp)], axis=1)
    datap1predi.columns = datap1pred.columns
    corr = datap1predi.corr()

    plt.figure(figsize=(20, 15))
    ax = sns.heatmap(
        corr,
        cmap="coolwarm",
        annot=True,
        fmt='.2g'
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right',
    );
    plt.tight_layout()



def svm1(request):
    # os.makedirs('/kaggle/working/results/')
    path = "../input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/train/NonDemented/"
    dirs = os.listdir(path)
    for item in dirs:
        if os.path.isfile(path + item):
            img = cv2.imread(path + item, cv2.IMREAD_COLOR)
            im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            im = cv2.resize(im, (190, 340))
            x, y, w, h = cv2.boundingRect(im)
            ret, thresh = cv2.threshold(im, 127, 255, 0)
            contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            M = cv2.moments(cnt)

            if int(M['m00']) == 0:
                cx = 0

            else:
                cx = int(M['m10'] / M['m00'])

            if int(M['m00']) == 0:
                cy = 0

            else:
                cy = int(M['m01'] / M['m00'])

            corners = cv2.goodFeaturesToTrack(im, 35, 0.01, 10)
            corners = np.int0(corners)
            X = []
            Y = []
            for i in corners:
                x, y = i.ravel()
                x = x - cx
                y = y - cy
                X.append(x)
                Y.append(y)

            tab = np.array([X, Y])
            tab = tab.T
            distance = []
            distance.append('NonDemented')
            for i in range(tab.shape[0]):
                dis = ((tab[i, 0]) ** 2 + (tab[i, 1]) ** 2) ** 0.5
                distance.append(dis)

            # tab = np.array(distance)

            data = pd.DataFrame(distance, columns=['NonDemented'])
            d = data.to_csv('/kaggle/working/results/' + item + '.csv', index=False)

    path = "../input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/test/NonDemented/"
    dirs = os.listdir(path)
    for item in dirs:
        if os.path.isfile(path + item):
            img = cv2.imread(path + item, cv2.IMREAD_COLOR)
            im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            im = cv2.resize(im, (190, 340))
            x, y, w, h = cv2.boundingRect(im)
            ret, thresh = cv2.threshold(im, 127, 255, 0)
            contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            M = cv2.moments(cnt)

            if int(M['m00']) == 0:
                cx = 0

            else:
                cx = int(M['m10'] / M['m00'])

            if int(M['m00']) == 0:
                cy = 0

            else:
                cy = int(M['m01'] / M['m00'])

            corners = cv2.goodFeaturesToTrack(im, 35, 0.01, 10)
            corners = np.int0(corners)
            X = []
            Y = []
            for i in corners:
                x, y = i.ravel()
                x = x - cx
                y = y - cy
                X.append(x)
                Y.append(y)

            tab = np.array([X, Y])
            tab = tab.T
            distance = []
            distance.append('NonDemented')
            for i in range(tab.shape[0]):
                dis = ((tab[i, 0]) ** 2 + (tab[i, 1]) ** 2) ** 0.5
                distance.append(dis)

            # tab = np.array(distance)

            data = pd.DataFrame(distance, columns=['NonDemented'])
            d = data.to_csv('/kaggle/working/results/' + item + '.csv', index=False)
    path = "../input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/train/VeryMildDemented/"
    dirs = os.listdir(path)
    for item in dirs:
        if os.path.isfile(path + item):
            img = cv2.imread(path + item, cv2.IMREAD_COLOR)
            im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            im = cv2.resize(im, (190, 340))
            x, y, w, h = cv2.boundingRect(im)
            ret, thresh = cv2.threshold(im, 127, 255, 0)
            contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            M = cv2.moments(cnt)

            if int(M['m00']) == 0:
                cx = 0

            else:
                cx = int(M['m10'] / M['m00'])

            if int(M['m00']) == 0:
                cy = 0

            else:
                cy = int(M['m01'] / M['m00'])

            corners = cv2.goodFeaturesToTrack(im, 35, 0.01, 10)
            corners = np.int0(corners)
            X = []
            Y = []
            for i in corners:
                x, y = i.ravel()
                x = x - cx
                y = y - cy
                X.append(x)
                Y.append(y)

            tab = np.array([X, Y])
            tab = tab.T
            distance = []
            distance.append('VeryMildDemented')
            for i in range(tab.shape[0]):
                dis = ((tab[i, 0]) ** 2 + (tab[i, 1]) ** 2) ** 0.5
                distance.append(dis)

            # tab = np.array(distance)

            data = pd.DataFrame(distance, columns=['VeryMildDemented'])
            d = data.to_csv('/kaggle/working/results/' + item + '.csv', index=False)

    path = "../input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/test/VeryMildDemented/"
    dirs = os.listdir(path)
    for item in dirs:
        if os.path.isfile(path + item):
            img = cv2.imread(path + item, cv2.IMREAD_COLOR)
            im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            im = cv2.resize(im, (190, 340))
            x, y, w, h = cv2.boundingRect(im)
            ret, thresh = cv2.threshold(im, 127, 255, 0)
            contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            M = cv2.moments(cnt)

            if int(M['m00']) == 0:
                cx = 0

            else:
                cx = int(M['m10'] / M['m00'])

            if int(M['m00']) == 0:
                cy = 0

            else:
                cy = int(M['m01'] / M['m00'])

            corners = cv2.goodFeaturesToTrack(im, 35, 0.01, 10)
            corners = np.int0(corners)
            X = []
            Y = []
            for i in corners:
                x, y = i.ravel()
                x = x - cx
                y = y - cy
                X.append(x)
                Y.append(y)

            tab = np.array([X, Y])
            tab = tab.T
            distance = []
            distance.append('VeryMildDemented')
            for i in range(tab.shape[0]):
                dis = ((tab[i, 0]) ** 2 + (tab[i, 1]) ** 2) ** 0.5
                distance.append(dis)

            # tab = np.array(distance)

            data = pd.DataFrame(distance, columns=['VeryMildDemented'])
            d = data.to_csv('/kaggle/working/results/' + item + '.csv', index=False)

    path = "../input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/train/ModerateDemented/"
    dirs = os.listdir(path)
    for item in dirs:
        if os.path.isfile(path + item):
            img = cv2.imread(path + item, cv2.IMREAD_COLOR)
            im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            im = cv2.resize(im, (190, 340))
            x, y, w, h = cv2.boundingRect(im)
            ret, thresh = cv2.threshold(im, 127, 255, 0)
            contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            M = cv2.moments(cnt)

            if int(M['m00']) == 0:
                cx = 0

            else:
                cx = int(M['m10'] / M['m00'])

            if int(M['m00']) == 0:
                cy = 0

            else:
                cy = int(M['m01'] / M['m00'])

            corners = cv2.goodFeaturesToTrack(im, 35, 0.01, 10)
            corners = np.int0(corners)
            X = []
            Y = []
            for i in corners:
                x, y = i.ravel()
                x = x - cx
                y = y - cy
                X.append(x)
                Y.append(y)

            tab = np.array([X, Y])
            tab = tab.T
            distance = []
            distance.append('ModerateDemented')
            for i in range(tab.shape[0]):
                dis = ((tab[i, 0]) ** 2 + (tab[i, 1]) ** 2) ** 0.5
                distance.append(dis)

            # tab = np.array(distance)

            data = pd.DataFrame(distance, columns=['ModerateDemented'])
            d = data.to_csv('/kaggle/working/results/' + item + '.csv', index=False)

    path = "../input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/test/ModerateDemented/"
    dirs = os.listdir(path)
    for item in dirs:
        if os.path.isfile(path + item):
            img = cv2.imread(path + item, cv2.IMREAD_COLOR)
            im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            im = cv2.resize(im, (190, 340))
            x, y, w, h = cv2.boundingRect(im)
            ret, thresh = cv2.threshold(im, 127, 255, 0)
            contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            M = cv2.moments(cnt)

            if int(M['m00']) == 0:
                cx = 0

            else:
                cx = int(M['m10'] / M['m00'])

            if int(M['m00']) == 0:
                cy = 0

            else:
                cy = int(M['m01'] / M['m00'])

            corners = cv2.goodFeaturesToTrack(im, 35, 0.01, 10)
            corners = np.int0(corners)
            X = []
            Y = []
            for i in corners:
                x, y = i.ravel()
                x = x - cx
                y = y - cy
                X.append(x)
                Y.append(y)

            tab = np.array([X, Y])
            tab = tab.T
            distance = []
            distance.append('ModerateDemented')
            for i in range(tab.shape[0]):
                dis = ((tab[i, 0]) ** 2 + (tab[i, 1]) ** 2) ** 0.5
                distance.append(dis)

            # tab = np.array(distance)

            data = pd.DataFrame(distance, columns=['ModerateDemented'])
            d = data.to_csv('/kaggle/working/results/' + item + '.csv', index=False)
    path = "../input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/train/MildDemented/"
    dirs = os.listdir(path)
    for item in dirs:
        if os.path.isfile(path + item):
            img = cv2.imread(path + item, cv2.IMREAD_COLOR)
            im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            im = cv2.resize(im, (190, 340))
            x, y, w, h = cv2.boundingRect(im)
            ret, thresh = cv2.threshold(im, 127, 255, 0)
            contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            M = cv2.moments(cnt)

            if int(M['m00']) == 0:
                cx = 0

            else:
                cx = int(M['m10'] / M['m00'])

            if int(M['m00']) == 0:
                cy = 0

            else:
                cy = int(M['m01'] / M['m00'])

            corners = cv2.goodFeaturesToTrack(im, 35, 0.01, 10)
            corners = np.int0(corners)
            X = []
            Y = []
            for i in corners:
                x, y = i.ravel()
                x = x - cx
                y = y - cy
                X.append(x)
                Y.append(y)

            tab = np.array([X, Y])
            tab = tab.T
            distance = []
            distance.append('MildDemented')
            for i in range(tab.shape[0]):
                dis = ((tab[i, 0]) ** 2 + (tab[i, 1]) ** 2) ** 0.5
                distance.append(dis)

            # tab = np.array(distance)

            data = pd.DataFrame(distance, columns=['MildDemented'])
            d = data.to_csv('/kaggle/working/results/' + item + '.csv', index=False)

    path = "../input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/test/MildDemented/"
    dirs = os.listdir(path)
    for item in dirs:
        if os.path.isfile(path + item):
            img = cv2.imread(path + item, cv2.IMREAD_COLOR)
            im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            im = cv2.resize(im, (190, 340))
            x, y, w, h = cv2.boundingRect(im)
            ret, thresh = cv2.threshold(im, 127, 255, 0)
            contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            M = cv2.moments(cnt)

            if int(M['m00']) == 0:
                cx = 0

            else:
                cx = int(M['m10'] / M['m00'])

            if int(M['m00']) == 0:
                cy = 0

            else:
                cy = int(M['m01'] / M['m00'])

            corners = cv2.goodFeaturesToTrack(im, 35, 0.01, 10)
            corners = np.int0(corners)
            X = []
            Y = []
            for i in corners:
                x, y = i.ravel()
                x = x - cx
                y = y - cy
                X.append(x)
                Y.append(y)

            tab = np.array([X, Y])
            tab = tab.T
            distance = []
            distance.append('MildDemented')
            for i in range(tab.shape[0]):
                dis = ((tab[i, 0]) ** 2 + (tab[i, 1]) ** 2) ** 0.5
                distance.append(dis)

            # tab = np.array(distance)

            data = pd.DataFrame(distance, columns=['MildDemented'])
            d = data.to_csv('/kaggle/working/results/' + item + '.csv', index=False)

    # Concatenate all  the vectors  of  features
    import glob


    path = '/kaggle/working/results/'

    os.chdir(path)
    extension = 'csv'
    # files= os.listdir(path)
    files = [i for i in glob.glob('*.{}'.format(extension))]
    result = pd.concat([pd.read_csv(path + f) for f in files], axis=1)
    d = result.to_csv('/kaggle/working/Features.csv')

    d = pd.read_csv('/kaggle/working/Features.csv')
    d.head()
    d = d.T
    d1 = d.to_csv('/kaggle/working/FeaturesSVM.csv')
    data = data.drop([0], axis=0)
    data = data.drop(['Unnamed: 0'], axis=1)
    data.head()
    # convert to cateogry dtype
    data['0'] = data['0'].astype('category')
    # convert to category codes
    data['0'] = data['0'].cat.codes
    continuous = ['1', '2', '3', '4', '5', '6', '7', '8', '9',
                  '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
                  '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
                  '30', '31', '32', '33', '34', '35']

    scaler = MinMaxScaler(feature_range=(0, 4))
    for var in continuous:
        data[var] = data[var].astype('float64')
        data[var] = scaler.fit_transform(data[var].values.reshape(-1, 1))
    data.head()
    X = data.drop('0', axis=1)
    y = data['0']
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    y_test
    X_test = X_test.dropna(axis='rows', how='any')
    X_train = X_train.dropna(axis='rows', how='any')
    from sklearn.svm import SVC
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    y_pred
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Accuracy: {}%".format(svclassifier.score(X_test, y_test) * 100))
    return render(request,'admin/svm1.html')


def svm11(request):
    df = pd.read_csv(r'C:\Users\Hi\Desktop\alzheimer project\3 Alzheimer Disease Prediction using Machine Learning Algorithms\Code\alzmierdisease\data_small.csv')
    print(df.head(5))

    plt.bar(df['PTID'],df['AGE'])
    plt.title('alzmierdisease')
    plt.show()

    plt.bar(df['SITE'],df['DXCHANGE'])
    plt.title('alzmierdisease')
    plt.show()

    plt.bar(df['SITE'],df['EcogSPDivatt_bl'])
    plt.title('alzmierdisease')
    plt.show()

    qs=patientModel.objects.all()
    return render(request,'admin/patientdetails.html',{"object":qs})