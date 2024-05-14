import matplotlib.pyplot as plt  
import numpy as np  
  
x = [1,2,3,4,5,10,30,50,75,100]
# precision
# y = [0.98,0.96,0.94,0.94,0.92,0.86,0.69,0.72,0.47,0.40]
# F1 Score
y = [0.9539994539994541,0.9675152882476381,0.962308657104418,0.9677947588061206,0.9538880859210942,0.9264640694766635,0.8843381751848267,0.8911304051753489, 0.80231177910394, 0.7638711395101172]

plt.plot(x, y)  
  
# 添加标题和轴标签  
plt.title('Imbalance ratio for F1 Score line graph')  
plt.xlabel('Imbalance Ratio')  
plt.ylabel('F1 Score')  
  
# 显示图形  
plt.show()