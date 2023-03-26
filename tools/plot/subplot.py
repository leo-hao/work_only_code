import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# 读取数据
# data1 = pd.read_csv('./1.csv',index_col=0)
# 或自动获取文件夹下的所有数据
path = r"data\\"
file = os.listdir(path)

fig = plt.figure(figsize = (7,5))    #figsize是图片的大小
# g = green,“-” = 实线，label = 图例的名称，一般要在名称前面加一个u
## 子图设置
ax1 = fig.add_subplot(1, 1, 1) # 子图
rect1 = [0.6, 0.25, 0.35, 0.35] # 子图位置，[左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）
axins = ax1.inset_axes(rect1)
# 设置想放大区域的横坐标范围
tx0 = 0
tx1 = 150
# 设置想放大区域的纵坐标范围
ty0 = 1000
ty1 = 2500
sx = [tx0,tx1,tx1,tx0,tx0]
sy = [ty0,ty0,ty1,ty1,ty0]
plt.plot(sx,sy,"purple")
axins.axis([tx0,tx1,ty0,ty1])  # 坐标范围

lab = ['1','2','3','4']
color = ['g','b','r']
for i in range(0,len(file)-1):
    data_csv=file[i]
    data = pd.read_csv(path+data_csv,index_col=0)
    # 横坐标Episode，纵坐标Loss
    x = data['Episodes']
    y = data['Loss']
    # 整体loss曲线
    plt.plot(x, y, color[i],label =lab[i])
    # 局部loss曲线
    axins.plot(x, y,color[i])

# 最后一个数据的后半截用虚线展示
data_csv = file[len(file)-1]
data = pd.read_csv(path+data_csv,index_col=0)
x_s = data[data['Episodes'] <= 100]['Episodes']
y_s = data[data['Episodes'] <= 100]['Loss']
x_x = data[data['Episodes'] > 100]['Episodes']
y_x = data[data['Episodes'] > 100]['Loss']

plt.plot(x_s, y_s,'y-', label =lab[len(file)-1])
plt.plot(x_x, y_x,'y--')  # 对于虚实要统一颜色，均设为y

axins.plot(x_s, y_s,'y-')
axins.plot(x_x, y_x,'y--')

plt.legend()  # 显示
# 坐标/标题设置
plt.xlabel(u'iters')
plt.ylabel(u'loss')
plt.title('Compare loss for different models in training')
