import pickle #加载训练好的模型
with open('allModel.pickle', 'rb') as file:
    allModel = pickle.load(file)
    labelEncoder = allModel['labelEncoder']
    tfidfVectorizer = allModel['tfidfVectorizer']
    logistic_model = allModel['logistic_model']
#使用训练好的模型，预测数据 X 的标签
#predict_y = logistic_model.predict(X)