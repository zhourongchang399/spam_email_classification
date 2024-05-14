import re
import pandas as pd
from collections import defaultdict 

def data_process():
    # 读取数据集
    dataset = pd.read_csv("full/index.csv",sep=' ',header=None)
    columns = ['label','content']
    dataset.columns = columns

    # 去除英文
    def clean_str(string):
        string = re.sub(r"[^\u4e00-\u9fff]", " ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip()

    dataset['label'] = dataset['label'].replace(['ham', 'spam'], [0, 1])

    # 清理数据  
    def clean_content(dataset):  
        for index, row in dataset.iterrows():  
            with open(row['content'][3:], 'r', encoding='gbk',errors='ignore') as file:  
            # 读取全部内容  
                content = file.read()
            cleaned_content = clean_str(content)  
            dataset.at[index, 'content'] = cleaned_content  
        return dataset  
    
    # 分词并更新数据集  
    dataset = clean_content(dataset)
    
    # 写入CSV文件  
    dataset.to_csv('dataset/cn_processed_data.csv', index=False, encoding='utf-8-sig')
