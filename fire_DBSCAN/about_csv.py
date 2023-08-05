import os
import csv
def restart ():
    name="sensor_data.csv"
    remove_csv(name)
    creat_csv(name)


def remove_csv (name):
      # 删除文件
    file_path = os.path.join(os.getcwd(), name)  # 文件路径
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"已删除文件: {file_path}")
    else:
        print(f"文件不存在: {file_path}")


def creat_csv (name):
    new_file_path = os.path.join(os.getcwd(), name)  # 新文件路径
    with open(new_file_path, "w") as file:
        print(f"已创建新文件: {new_file_path}")


def truncate_csv():
    #script_path = os.path.dirname(__file__)
    #filename = os.path.join(script_path, 'sensor_data.csv')
    filename='dataset.csv'
    # Check if the file exists
    if not os.path.isfile(filename):
        print(f"The file {filename} does not exist.")
        return
    

# 使用 'w' 模式打开文件，并立即关闭，这样会清空文件中的内容
    with open(filename, 'w', newline='') as csvfile:
        pass
    # Open the file in write mode and truncate its contents
        print(f"文件 {filename} 已被清空.")
        #writer.writerow(['Temperature', 'Gas', 'Relative Humidity'])

    
def clean_final_dataset ():
    filename='final_dataset.csv'
    if not os.path.isfile(filename):
        print(f"The file {filename} does not exist.")
        return
    with open(filename, 'w', newline='') as csvfile:
        pass
    # Open the file in write mode and truncate its contents
        print(f"文件 {filename} 已被清空.")
