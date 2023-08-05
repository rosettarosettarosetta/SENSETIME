# -*- coding: utf-8 -*- 
# /*
#                    _ooOoo_
#                   o8888888o
#                   88" . "88
#                   (| -_- |)
#                   O\  =  /O
#                ____/`---'\____
#              .'  \\|     |//  `.
#             /  \\|||  :  |||//  \
#            /  _||||| -:- |||||-  \
#            |   | \\\  -  /// |   |
#            | \_|  ''\---/''  |   |
#            \  .-\__  `-`  ___/-. /
#          ___`. .'  /--.--\  `. . __
#       ."" '<  `.___\_<|>_/___.'  >'"".
#      | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#      \  \ `-.   \_ __\ /__ _/   .-` /  /
# ======`-.____`-.___\_____/___.-`____.-'======
#                    `=---='
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#             佛祖保佑       永无BUG
# */

from se_openhw.kit.nano.hat import EnvironmentalSensor
#from STESDK import HandDetector
import time
import os  
from DBSCAN import IncrementalDBSCAN 
import csv
import datetime
import pandas as pd
from sense.about_csv import truncate_csv ,clean_final_dataset
from show import show_3d

# 检查数据是否合法
def check_data_valid(temperature, gas, relative_humidity, pressure, altitude):
    if temperature == 0 or gas == 0 or relative_humidity == 0 or pressure == 0 or altitude == 0:
      raise ValueError("传感器读取数据存在异常！")


def count_csv_rows(file_path):  #查看当前数据量
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        row_count = sum(1 for row in csvreader)
        print('现在拥有数据'+str(row_count))
    return row_count


sensor=EnvironmentalSensor()
temperature = gas = relative_humidity = pressure = altitude = 0
dbscan = IncrementalDBSCAN()
show_3D=show_3d()


#script_path = os.path.dirname(__file__)
#filename = os.path.join(script_path, 'sensor_data.csv')
filename=os.path.join('dataset.csv')
if os.path.isfile(filename):
    print(f"The file {filename} exists.")
else:
    print(f"The file {filename} does not exist.")
clean_final_dataset()




total_rows = count_csv_rows(filename)

if total_rows != 0:
  print("检测到有数据保存，是否调用曾经的数据数据 [y]/[n]  （如果更换环境，建议重置）")

  user_input = input().lower()  # 将用户输入转换为小写字母，以便不区分大小写
  if user_input == 'y'or user_input == '':
    print("调用数据中")
    dbscan.set_data(filename)
    # 在这里执行重置数据的操作
        
  elif user_input == 'n':
    print("重置数据中")
    # 在这里执行不重置数据的操作
    truncate_csv ()
    total_rows=-1
    print("重置完成")
        


# 打开 CSV 文件并创建 csv.writer 对象
with open(filename, 'a', newline='') as csvfile:
  csvwriter = csv.writer(csvfile)
  while True:
    
    # 模拟从传感器读取的数据
    temperature, gas, relative_humidity, pressure, altitude = sensor.read()
    temperature, gas, relative_humidity, pressure, altitude = map(lambda x: round(float(x), 2), sensor.read())
    total_rows=total_rows+1 #第几行

    try:
      check_data_valid(temperature, gas, relative_humidity, pressure, altitude)
    except ValueError as e:
      print(f"发现错误：{e}")
      continue
    
    
    #timestamp = datetime.datetime.now()
    # 将数据写入 CSV 文件
  
    print(total_rows )
    csvwriter.writerow([temperature, pressure, relative_humidity])#
    csvfile.flush() 
    
     
    #if total_rows < 20 and total_rows>0:
    if total_rows < 40 :
      dbscan.set_data(filename)
      dbscan.batch_dbscan()
      
      show_3D.batch_dbscan_3d(dbscan)
      #clean_final_dataset()
      #csvfile.flush() 
      dbscan.print_final_dataset
      print(dbscan.final_dataset.values)
    if total_rows>=40:
      dbscan.set_data(filename)
      dbscan.incremental_dbscan_()
      #dbscan.add_dataset(temperature, pressure, relative_humidity)#
    csvfile.flush() 
    #time.sleep(3)
