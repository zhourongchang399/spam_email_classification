from sklearn.model_selection import train_test_split
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.naive_bayes import MultinomialNB,GaussianNB  
import re, argparse
from utils.result_process import conclusion
from nltk.tokenize import RegexpTokenizer
import nltk
from collections import defaultdict 
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default='en_email', help='dataset')
parser.add_argument("--amount", type=int, default=2000, help='the number of example')
parser.add_argument("--seed", type=int, default=23, help='Random seed')
parser.add_argument("--size", type=float, default=0.2, help='test_size')
parser.add_argument("--tfidf", action='store_true', help='tfidf')
parser.add_argument("--alpha", type=float, default=1.0, help='Laplace Smoothing')

args = parser.parse_args()

# 读取路径和停用词
if args.dataset == 'cn_email':
    path = "dataset/cn_processed_data.csv"
    stop_words_datas = pd.read_csv("dataset/cn_stop_words.csv",header=None)  
else:
    path = "dataset/spam_ham_dataset.csv"
    stop_words_datas = pd.read_csv("dataset/en_stop_words.csv",header=None)  

stop_words_list = stop_words_datas.iloc[:,0].tolist()  

# 读取数据
df = pd.read_csv(path, nrows=args.amount)  
rows_with_nan = df.isnull().any(axis=1)  

# 去除无法读取的例子 
dataset = df[~rows_with_nan]  

if args.dataset == 'cn_email':
    # 中文分词
    dataset['tokenized_content'] = dataset['content'].apply(lambda text: list(jieba.cut(text, cut_all=False)))  
    # 分词拼接为一个字符串
    dataset['tokenized_content_str'] = dataset['tokenized_content'].apply(lambda tokens: ' '.join(tokens))  
else:
    try:
        # 查找模型
        nltk.data.find('tokenizers/punkt')
    except LookupError:  
        # 如果未找到，则下载punkt模型
        nltk.download('punkt')

    # 将文本转换为小写
    dataset['content'] = dataset['content'].str.lower()  
    tokenizer = RegexpTokenizer(r'\w+')
    # 英文分词
    dataset['tokenized_content'] = dataset['content'].apply(lambda text: tokenizer.tokenize(text))
    # 分词拼接为一个字符串
    dataset['tokenized_content_str'] = dataset['tokenized_content'].apply(lambda tokens: ' '.join(tokens))

if args.tfidf:
    # 使用TfidfVectorizer将文本转换为TF-IDF特征向量  
    vectorizer = TfidfVectorizer(stop_words=stop_words_list)  
    X = vectorizer.fit_transform(dataset['tokenized_content_str'])
    print(X.shape)

else:
    stop_words_set = set(stop_words_list)
    # 建立词库  
    word_counts = defaultdict(int)  
    for tokens in dataset['tokenized_content']:  
        for token in tokens:  
            if token not in stop_words_set:  # 假设stop_words_set是停用词集合  
                word_counts[token] += 1  
    
    # 将词映射到索引（建立词库到索引的映射）  
    vocabulary = {word: idx for idx, word in enumerate(word_counts.keys(), start=0)}  
    
    # 构建词频矩阵  
    X = []  
    for tokens in dataset['tokenized_content']:  
        feature_vector = [0] * len(vocabulary)  # 初始化词频向量  
        for token in tokens:  
            if token in vocabulary:  
                feature_vector[vocabulary[token]] += 1   
        X.append(feature_vector)  
    
    X = np.array(X)   
    print(X.shape)

# 划分测试训练数据集
Y = dataset['label']
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    test_size=args.size,
                                                    random_state=args.seed)
  
# 初始化并训练朴素贝叶斯分类器  
clf = MultinomialNB(alpha=args.alpha)  
clf.fit(X_train, y_train)  
  
# 预测测试集  
y_pred = clf.predict(X_test)  

conclusion(y_test, y_pred)