import os
import jieba
with open('D:/机器学习经典案例/垃圾邮件分类/垃圾邮件分类/full/index') as file:
    y = [k.split()[0] for k in file.readlines()]
    print(len(y))

from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
y_encode = labelEncoder.fit_transform(y)

def getFilePathList2(rootDir):
    filePath_list = []
    for walk in os.walk(rootDir):
        part_filePath_list = [os.path.join(walk[0], file) for file in walk[2]]
        filePath_list.extend(part_filePath_list)
    return filePath_list

filePath_list = getFilePathList2('D:/机器学习经典案例/垃圾邮件分类/垃圾邮件分类/data')
mailContent_list = []
for filePath in filePath_list:
    with open(filePath, errors='ignore') as file:
        file_str = file.read()
        mailContent = file_str.split('\n\n', maxsplit=1)[1] #只保留正文部分的内容
        mailContent_list.append(mailContent)
print(mailContent_list[1])

import re
mailContent_list = [re.sub('\s+', ' ', k) for k in mailContent_list]
with open('D:/机器学习经典案例/垃圾邮件分类/垃圾邮件分类/stopwords.txt', encoding='utf8') as file:
    file_str = file.read()
    stopword_list = file_str.split('\n')
    stopword_set = set(stopword_list)

import time
cutWords_list = []
startTime = time.time()
for mail in mailContent_list:
    cutWords = [k for k in jieba.lcut(mail) if k not in stopword_set]
    cutWords_list.append(cutWords)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(cutWords_list, min_df=100, max_df=0.25)
X = tfidf.fit_transform(mailContent_list)
#模型训练
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y_encode, test_size=0.2)
from sklearn.linear_model import LogisticRegressionCV
logistic_model = LogisticRegressionCV()#初始化一个逻辑回归模型
logistic_model.fit(train_X, train_y) #调用 sklearn 的模型训练程序
sorce = logistic_model.score(test_X, test_y)#计算模型在测试数据上的分类正确率
print(logistic_model.Cs)
import pickle #将模型参数、数值类别对应的真实标签保存在文件 allModel.pickle 中
with open('allModel.pickle', 'wb') as file:
    save = {
        'labelEncoder' : labelEncoder,
        'tfidfVectorizer' : tfidf,
        'logistic_model' : logistic_model
    }
    pickle.dump(save, file)

