from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix  

def conclusion(y_test, y_pred):
    # 打印分类报告  
    print(classification_report(y_test, y_pred))  
    
    # 打印准确度  
    accuracy = accuracy_score(y_test, y_pred)  
    print("Accuracy:", accuracy)  
    
    # 假设是多分类问题，计算平均的精确率、召回率和F1分数  
    precision = precision_score(y_test, y_pred, average='macro')  # 或者 'weighted', 'micro'  
    recall = recall_score(y_test, y_pred, average='macro')  
    f1 = f1_score(y_test, y_pred, average='macro')  
    
    print("Precision:", precision)  
    print("Recall:", recall)  
    print("F1 Score:", f1)  
    
    # 打印混淆矩阵  
    cm = confusion_matrix(y_test, y_pred)  
    print("Confusion Matrix:")  
    print(cm)