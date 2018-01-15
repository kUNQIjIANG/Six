## This script contains a logistic multi-classes regression 
## where take description as input and activity as output.

import pyodbc
import numpy as np
import random
from collections import defaultdict
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier

db_conn = pyodbc.connect(
    'Driver={SQL Server};Server=ccdc1sql236\sql_dtlab_dev1;DATABASE=DataLab;trusted_connection=yes')  # only need [] around the table name if there are space characters in it, otherwise it's optional.


cursor = db_conn.cursor()

cursor.execute('select Narrative, ActivityGroup from [Narrative_Activities] \
inner join [3EMatterExtract] ON Narrative_Activities.ActivityDescription = [3EMatterExtract].ActivityDescription')


narrative = []
activity = []
data_units = 3000

def checkNum(dic, num):
    check = True
    if len(dic) == 0:
        check = False
    for key, value in dic.items():
        if len(dic[key]) < num:
            check = False
    return check

colle = defaultdict(list)
for row in cursor:
    if row[0] is not None:
        if len(colle[row[1]]) < data_units:
            colle[row[1]].append(row[0])
    if checkNum(colle,data_units):
        break
print('collect done')


for key in colle.keys():
    for nar in colle[key]:
        activity.append(key)
        narrative.append(nar)

print('add done')

def featurize(narr):
    clean_word = [word for word in str(narr).split(' ') if word not in stopwords.words('english')]
    word_dict = defaultdict(int)
    for word in clean_word:
        word_dict[word] += 10
    return word_dict

def get_data(text, label, vol):
    t = np.array(text[:vol]).reshape(vol,1)
    l = np.array(label[:vol]).reshape(vol,1)
    c = list(np.hstack((t,l)))
    random.shuffle(c)
    train = c[:len(c)//4 * 3]
    test = c[len(c)// 4 * 3 :]
    return train, test

# Trainning the model
def logisticRegression(train_set):
    vectorizer = DictVectorizer()
    label_encoder = LabelEncoder()
    train_narr = vectorizer.fit_transform([featurize(x) for x,_ in train_set])
    train_act = label_encoder.fit_transform([y for _,y in train_set])

    lr = LogisticRegression(C=0.1)
    lr.fit(train_narr, train_act)
    return lr, vectorizer, label_encoder


def neuralNet(train_set):
    vectorizer = DictVectorizer()
    label_encoder = LabelEncoder()
    train_narr = vectorizer.fit_transform([featurize(x) for x,_ in train_set])
    train_act = label_encoder.fit_transform([y for _,y in train_set])

    nn = MLPClassifier(hidden_layer_sizes = (10,3), batch_size = 50, alpha = 0.01)
    nn.fit(train_narr, train_act)
    return nn, vectorizer, label_encoder

def classDistribute(label_list):
    count = defaultdict(int)
    for a in label_list:
        count[a] += 1
    print(count)

def classDistribute(label_list):
    count = defaultdict(int)
    for a in label_list:
        count[a] += 1
    print(count)

use_narrative = narrative
use_activity = activity
train_set, test_set = get_data(use_narrative, use_activity, len(use_narrative))
print('get data done')
#nn, vectorizer, label_encoder = neuralNet(train_set)
lr, vectorizer, label_encoder = logisticRegression(train_set)
print('fitting done')
test_set_x = vectorizer.transform([featurize(x) for x,_ in test_set])
test_set_y = [y for _,y in test_set]
train_set_y = [y for _,y in train_set]

print(classDistribute(train_set_y))
print(classDistribute(test_set_y))


y_pred = label_encoder.inverse_transform(lr.predict(test_set_x))
#y_pred = label_encoder.inverse_transform(nn.predict(test_set_x))


print(confusion_matrix(test_set_y, y_pred))


print(f1_score(test_set_y, y_pred, average='micro'))
print(f1_score(test_set_y, y_pred, average='macro'))