import matplotlib.pyplot as plt
import numpy as np
import csv

x1 = []  # x1
x2 = []  # x2
y = []
i = 1
with open('saved_models/cave/gamma_ftall/randga95_im_a8/exp_0/logger.txt','r') as fr:
    lines = fr.readlines()
    for line in lines:
        items = line.strip().split('|')
        xx = items[-1].strip().split(':')[-1]

        yy = items[-3].strip().split(':')[-1].strip().split('e')[0]
        xx = float(xx)
        yy = float(yy)
        x1.append(xx)
        x2.append(yy)
        y.append(i)
        i = i + 1
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(y,x1,'r--', label='PSNR')
ax2.plot(y,x2,'b--', label='Loss')
ax1.set_ylim(18, 36)
ax2.set_ylim(5.4, 8)
## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')


ax1.set_xlabel('Epoch')
ax1.set_ylabel('PSNR(dB)',color='r')
ax2.set_ylabel('Loss(1e-1)',color='b')


# plt.ylim(-1,1)#仅设置y轴坐标范围
plt.savefig('./test.jpg')
plt.show()
