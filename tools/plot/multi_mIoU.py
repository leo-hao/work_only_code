import os
import pandas as pd
import matplotlib.pyplot as plt
'''
os.listdir(folder_path)函数用于返回文件夹中的文件列表，
filename.endswith(".csv")用于筛选出以".csv"结尾的文件；
pd.read_csv(file_path)函数用于读取CSV文件数据；
ax.plot(df["x"], df["y"], label=filename[:-4])
函数用于绘制曲线并设置图例，filename[:-4]表示去掉文件名的后缀".csv"，作为该曲线的图例名称；
ax.legend()函数用于设置图例并显示图像。


'''
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置文件夹路径
folder_path = "/path/to/folder"

# 读取文件夹中所有CSV文件数据并绘制曲线图
fig, ax = plt.subplots()

# 设置颜色调色板
color_palette = sns.color_palette("husl"l, n_colors=len(os.listdir(folder_path)))

for i, filename in enumerate(os.listdir(folder_path)):
    if filename.endswith(".csv"):
        # 读取CSV文件数据
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        
        # 绘制曲线图
        ax.plot(df["x"], df["y"], label=filename[:-4], color=color_palette[i])

# 设置图例并显示图像
#bbox_to_anchor表示图例框的左上角放在哪里，(1.05, 1)表示放在图像的右上角略微向外偏移，loc表示图例框中的标签放在哪里，'upper left'表示左上角。
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
