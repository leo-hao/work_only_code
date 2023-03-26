import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取csv文件并存储数据到列表中
data = []
for filename in ['file1.csv', 'file2.csv', 'file3.csv']:
    df = pd.read_csv(filename)
    data.append(df)


# 绘制多条曲线
fig, ax = plt.subplots()
for i, df in enumerate(data):
    x = df['x'].values
    y = df['y'].values
    ax.plot(x, y, label=f'Curve {i+1}')

# 添加图例
ax.legend()

# 显示图形
plt.show()


# 绘制图片
...

# 保存图片
plt.savefig('filename.png')
