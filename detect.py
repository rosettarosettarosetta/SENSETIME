import subprocess
import threading
#ubprocess.run("conda deactivate", shell=True)

# 定义要运行的命令行指令
command1 = "python3 detect_yolo.py"

def run_command1():
    subprocess.run("cd apply", shell=True)
    command1 = "python3 /home/senseedu/sensetime/apply/detect_yolo.py"
    subprocess.run(command1,shell=True)

def run_command2():
    #subprocess.run("cd fire_DBSCAN", shell=True)
    command2 = "python3 /home/senseedu/sensetime/fire_DBSCAN/detect_sensor.py"
    subprocess.run(command2, shell=True)

# def run_command3 ():
#     command3="python3 /home/senseedu/sensetime/apply/camera.py"
#     subprocess.run(command3,shell=True)

# 创建线程
thread1 = threading.Thread(target=run_command1)
thread2 = threading.Thread(target=run_command2)
# thread3 = threading.Thread(target=run_command3)

# 启动线程
thread1.start()
thread2.start()
# thread3.start()

# 等待线程结束

print("三个指令行执行开始")
