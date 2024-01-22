import numpy as np
import pandas as pd

# 读取名为"metric"的表格数据
data = pd.read_excel('../result/metric.xlsx')  # 你需要根据你的数据格式来读取表格数据

# 计算每列的均值
mean_values = data.mean()

# 计算每列的标准差
std_values = data.std()

# 打印均值和标准差
print("均值:")
print(mean_values)
print("\n标准差:")
print(std_values)