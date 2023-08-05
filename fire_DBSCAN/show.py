import threading
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from DBSCAN import IncrementalDBSCAN
import queue
import random



class show_3d ():
    def __init__(self):
        # 创建三维坐标轴
        self.fig = plt.figure()
        self.ax =  self.fig.add_subplot(111, projection='3d')
        self.data_points = {}
        self.colors = {}
        self.colors[-1] = '#FF0000'
        self.ax.set_xlabel('Temperature')
        self.ax.set_ylabel('Gas')
        self.ax.set_zlabel('Relative Humidity')
        self.ax.set_xlim(20, 35)
        self.ax.set_ylim(990, 1005)
        self.ax.set_zlim(50, 70)
        self.label=-1

        self.ax.legend()


    def add_color(self,dbscan_instance):
        while dbscan_instance.Label != self.label :
            self.label=self.label+1
            random_color = None
            while random_color is None or random_color in self.colors.values():
                random_color = '#' + ''.join(random.choices('0123456789ABCDEF', k=6))
            self.colors[self.label]=random_color

            for key, value in self.colors.items():
                print(key, ":", value)
        
        
        


    def batch_dbscan_3d(self,dbscan_instance):

        self.add_color(dbscan_instance)

        for index, row in dbscan_instance.final_dataset.iterrows():
            temperature = row['Temperature']
            gas = row['Gas']
            relative_humidity = row['Relative Humidity']
            label = row['Label']
            
    
            self.ax.scatter(temperature, gas, relative_humidity, color=self.colors[label], label=f'Label {label}') 
            

            #for label, row in dbscan_instance.final_dataset.iterrows():
            #    color = self.colors[row['Label']]
                     # 绘制数据点
            #self.ax.scatter(temperature, gas, relative_humidity, color=color, label=f'Label {label}')   
        plt.draw()  # 添加图例
            # 暂停一小段时间，使图形得以更新
                #aplt.show()
        plt.pause(3)  
        self.ax.clear()  # 清空之前绘画的点
