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

def imbalance_process(df : pd.DataFrame, ir : float):
    maj = 0
    min = 1

    if df['label'].value_counts()[maj] < df['label'].value_counts()[min]:
        min = 0
        maj = 1

    # 计算当前不平衡率  
    minority_count = df['label'].value_counts()[min]  # 少数类的数量  
    majority_count = df['label'].value_counts()[maj]  # 多数类的数量  
    current_imbalance_ratio = majority_count / minority_count  
    print(f"Current minority_count: {minority_count}")
    print(f"Current majority_count: {majority_count}")
    print(f"Current imbalance ratio: {current_imbalance_ratio}")  
    
    # 设置目标不平衡率  
    target_imbalance_ratio = ir
    
    # 如果当前不平衡率高于目标不平衡率，则减少多数类样本  
    if current_imbalance_ratio > target_imbalance_ratio:  
        # 计算需要剔除的多数类样本数量  
        samples_to_remove = int((current_imbalance_ratio - target_imbalance_ratio) * minority_count)  
        # 从多数类中随机选择样本并剔除  
        df_majority = df[df['label'] == maj]  
        df_majority_sampled = df_majority.sample(n=len(df_majority) - samples_to_remove, random_state=42)  
        # 合并剔除样本后的多数类与少数类  
        df_sampled = pd.concat([df_majority_sampled, df[df['label'] == min]])  
    else:
        samples_to_remove = int(minority_count - (majority_count // target_imbalance_ratio))  
        if samples_to_remove < 0:  
            # 如果已经足够接近或超过目标不平衡率，则不需要剔除样本  
            samples_to_remove = 0  
        
        # 从少数类中随机选择样本并剔除  
        df_minority = df[df['label'] == min]  
        df_minority_sampled = df_minority.sample(n=len(df_minority) - samples_to_remove, random_state=42)  
        
        # 合并剔除样本后的少数类与多数类  
        df_sampled = pd.concat([df_minority_sampled, df[df['label'] == maj]])  

    minority_count_new = df_sampled['label'].value_counts()[min]  
    majority_count_new = df_sampled['label'].value_counts()[maj]  
    new_imbalance_ratio = majority_count_new / minority_count_new  
    print(f"Current minority_count: {minority_count_new}")
    print(f"Current majority_count: {majority_count_new}")
    print(f"Current imbalance ratio: {new_imbalance_ratio}")  
        
    dataset = df_sampled
    dataset['content'] = dataset['content'].astype(str)  

    return dataset