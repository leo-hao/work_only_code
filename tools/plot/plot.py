import csv
import matplotlib.pyplot as plt
import numpy as np

# 读取三个csv文件
file_names = ["result1.csv", "result2.csv", "result3.csv"]
data = []
for file_name in file_names:
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
        data.append(rows[1:])  # 跳过第一行类别名称，只取IoU数据

# 提取类别名称和IoU数据
class_names = [row[0] for row in data[0]]  # 假设三个csv文件中的类别名称都是一样的，只取其中一个文件的类别名称即可
iou_data = []
for i in range(len(data)):
    iou_data.append([float(row[1]) for row in data[i]])

# 绘制柱状图
fig, ax = plt.subplots()
bar_width = 0.2
opacity = 0.8
colors = ['r', 'g', 'b']
for i in range(len(iou_data)):
    x = np.arange(len(class_names)) + i * bar_width
    rects = ax.bar(x, iou_data[i], bar_width,
                    alpha=opacity, color=colors[i % len(colors)],
                    label="Result " + str(i+1))

ax.set_xlabel('Class')
ax.set_ylabel('IoU')
ax.set_title('IoU Results')
ax.set_xticks(np.arange(len(class_names)) + bar_width)
ax.set_xticklabels(class_names)
ax.legend()
plt.show()
