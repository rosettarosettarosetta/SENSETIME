import pandas as pd
import io
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading
import queue
#import plotly.graph_objects as go



def distance(element_1, element_2):
    """
    Calculates the distance between the element and the mean_core_element using the Euclidean distance
    :param element_1:  the current element that needs to be checked
    :param element_2:  the element to check the distance from
    :returns distance: the Euclidean distance between the element_1 and the element_2(float)
    """
    euclidean_distance = ((element_1['Temperature'] - element_2['Temperature']) ** 2 +
                          (element_1['Gas'] - element_2['Gas']) ** 2 +
                          (element_1['Relative Humidity'] - element_2['Relative Humidity']) ** 2) ** (1 / 2)
    return euclidean_distance.iloc[0].astype(float)


class IncrementalDBSCAN:
    
    def __init__(self, eps=0.3, min_samples=3):
        """
        Constructor the Incremental_DBSCAN class.
        :param eps:  the  maximum radius that an element should be in order to formulate a cluster
        :param min_samples:  the minimum samples required in order to formulate a cluster
        In order to identify the optimum eps and min_samples we need to  make a KNN
        """
        self.dataset = pd.DataFrame(columns=['Temperature', 'Gas', 'Relative Humidity'])
        self.labels = pd.DataFrame(columns=['Label'])
        self.final_dataset = pd.DataFrame(columns=['Temperature', 'Gas', 'Relative Humidity', 'Label'])  #存储聚类后的最终数据
        self.mean_core_elements = pd.DataFrame(columns=['Temperature', 'Gas', 'Relative Humidity', 'Label']) #核心点的平均值数据
        self.eps = eps
        self.min_samples = min_samples
        self.largest_cluster = -1   #这个属性将用于存储最大的聚类标签，初始值为 -1，表示还没有进行聚类。
        self.cluster_limits = 0  #聚类的数量
        self.largest_cluster_limits = 0  #数据集中最大的聚类簇包含的样本数
        self.Label = -1
        #self.show_3d_init()
        #self.plot_thread = threading.Thread(target=self.show_plot)
        #self.plot_thread.start()

        #self.queue = queue.Queue() 

        #from show import TestThread
        #self.plot_thread = TestThread(dbscan_instance=self)
        #self.plot_thread.start()
       

    
    
        



    def set_data(self, message):
        """
        After the connection with the RabbitMQ is complete a message is received.
        This function is used to gather the message from the consumer. It appends the newly arrived data to the
        dataset used for clustering.
        :param message:  The message consumed by the RabbitMQ. Should be a 3-column, comma-separated text.
        """
        # store the collected message to a temp dataframe
        #temp = pd.read_csv(io.StringIO(message), sep=',', header=None)
        #temp.columns = ['Temperature', 'Gas', 'Relative Humidity']
        # append the temp to the dataset
        data_frame=pd.read_csv(message,sep=',',header=None)
        data_frame.columns=['Temperature', 'Gas', 'Relative Humidity']
        last_row = data_frame.iloc[-1] 
        self.dataset = self.dataset.append(last_row, ignore_index=True)

        

    def print_final_dataset(self):
        print(self.final_dataset.values)
    def batch_dbscan(self):
        """
        The DBSCAN algorithm taken from the sklearn library. It is used to formulate the clusters the first time.
        Based on the outcomes of this algorithm the Incremental_DBSCAN algorithm
        """
        batch_dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(self.dataset)
        # Get the number of the clusters created
        # n_clusters_ = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        self.add_labels_to_dataset(batch_dbscan.labels_)

        # Cast everything imax_label_value
        # if response is not None:
        #     self.check_min_samples_in_eps_or_outlier(min_dist_index=response)
        # self.largest_cluster = self.find_largest_cluster()
        #self.draw_3d()
        #self.queue.put(1)
        self.lagest_labels()
       
        


    def add_labels_to_dataset(self, labels): #聚类标签添加到原始数据集中，并将其存储在final_dataset属性中
        """
        This function adds the labels on the dataset after the batch DBSCAN is done
        :param labels: The labels param should be a list that  describes the cluster of each element.
        If an element is considered as an outlier it should be equal to -1  如果一个数据点被视为噪声点或异常点，则标签值为-1。
        """
        self.labels = pd.DataFrame(labels, columns=['Label'])
        self.final_dataset = pd.concat([self.dataset, self.labels], axis=1)
        self.final_dataset.tail(n=1).to_csv('final_dataset.csv', mode='a', header=False, index=False)
        #csvfile.flush() 
        #self.final_dataset.to_csv('final_dataset.csv', index=False, header=False)  #添加到csv（全部）

    def sort_dataset_based_on_labels(self):  #同标签的在一起  #暂时不用这个
        """
        This function sorts the dataset based on the Label of each cluster.
        """
        # print(self.final_dataset)
        self.final_dataset = self.final_dataset.sort_values(by=['Label'])  #照聚类标签'Label'对数据集进行排序
        # Cast everything in the final_dataset as integer.
        # If this line is missing, it throws an error
        #self.final_dataset = self.final_dataset.astype(int)

    def lagest_labels (self):
        max_label_value = self.final_dataset['Label'].max()
        if max_label_value>self.Label:
            self.Label=max_label_value
    

    def find_mean_core_element(self):  #核心平均
        """
        This function calculates the average core elements of each cluster.
        Note: It does not calculate an average core element for the outliers.
        """
        # Exclude rows labeled as outliers
        self.mean_core_elements = self.final_dataset.loc[self.final_dataset['Label'] != -1]
        # Find the mean core elements of each cluster
        self.mean_core_elements = self.mean_core_elements \
            .groupby('Label')['Temperature', 'Gas', 'Relative Humidity'].mean()
        # print(self.mean_core_elements)
        # response = self.calculate_min_distance_centroid()
        # # print(response)
        # if response is not None:
        #     self.check_min_samples_in_eps_or_outlier(min_dist_index=response)

    def calculate_min_distance_centroid(self):  #用于计算新元素与每个聚类簇的核心元素之间的距离，并找到最近的
        """
        This function identifies the closest mean_core_element to the incoming element
        that has not yet been added to a cluster or considered as outlier.
        The distance is calculated using the distance function as it is described above.

        :returns min_dist_index: if there is a cluster that is closest to the new entry element
        or None if there are no clusters yet.
        """
        min_dist = None
        min_dist_index = None

        # Check if there are elements in the core_elements dataframe.
        # In other words if there are clusters created by the DBSCAN algorithm
        if not self.mean_core_elements.empty:           #是否已经存在聚类簇
            # Iterate over the mean_core_elements dataframe and find the minimum distance
            for index, current_mean_core_element in self.mean_core_elements.iterrows():
                tmp_dist = distance(element_1=self.final_dataset.tail(n=1), #当前元素（最后一行）的数据
                                    element_2=current_mean_core_element)
                if min_dist is None:
                    min_dist = tmp_dist
                    min_dist_index = index
                elif tmp_dist < min_dist:
                    min_dist = tmp_dist
                    min_dist_index = index
            print('Minimum distance is: ', min_dist, ' at cluster ', min_dist_index) #改
            return min_dist_index
        else:
            return None

    def check_min_samples_in_eps_or_outlier(self, min_dist_index):
        """
        This function checks whether there are at least min_samples in the given radius from the new
        entry element.
        If there are at least min_samples this element will be added to the cluster and the
        mean_core_element of the current cluster has to be re-calculated.
        If not, there are two options.
            1. Check if there are at least min_samples  outliers in the given radius in order to create a new
                cluster, or
            2.  Consider it as a new outlier

        :param min_dist_index: This is the parameter that contains information related to the closest
        mean_core_element to the current element.
        """

        # Use only the elements of the closest cluster from the new entry element
        new_element = self.final_dataset.tail(1)
        nearest_cluster_elements = self.final_dataset[self.final_dataset['Label'] == min_dist_index]
        min_samples_count = 0
        for index, cluster_element in nearest_cluster_elements.iterrows():
            if (cluster_element['Temperature'] - self.eps
                <= float(new_element['Temperature']) <= cluster_element['Temperature'] + self.eps) \
                    and (cluster_element['Gas'] - self.eps
                         <= float(new_element['Gas']) <= cluster_element['Gas'] + self.eps) \
                    and (cluster_element['Relative Humidity'] - self.eps
                         <= float(new_element['Relative Humidity']) <= cluster_element['Relative Humidity'] + self.eps):
                min_samples_count += 1

        if min_samples_count >= self.min_samples:
            # The new element  has enough cluster labels in the eps range
            #  and is now considered as a new member of the cluster.
            #  The mean core element of this cluster is re-calculated.
            self.final_dataset.loc[self.final_dataset.index[-1], 'Label'] = min_dist_index
            self.find_mean_core_element()
        else:
            # The new element is not added to its closest cluster. Now we have to check
            # whether it is going to be considered an outlier or it will form a new cluster
            # with other outliers.
            outliers = self.final_dataset[self.final_dataset['Label'] == -1]
            min_outliers_count = 0
            new_cluster_elements = pd.DataFrame(columns=['Index'])
            for index, outlier in outliers.iterrows():
                if (outlier['Temperature'] - self.eps
                    <= float(new_element['Temperature']) <= outlier['Temperature'] + self.eps) \
                        and (outlier['Gas'] - self.eps
                             <= float(new_element['Gas']) <= outlier['Gas'] + self.eps) \
                        and (outlier['Relative Humidity'] - self.eps
                             <= float(new_element['Relative Humidity']) <= outlier['Relative Humidity'] + self.eps):
                    min_outliers_count += 1
                    new_cluster_elements = new_cluster_elements.append({"Index": index}, ignore_index=True)

            if min_outliers_count >= self.min_samples:
                # The new element has enough outliers in its eps radius in order to form a new cluster.
                new_cluster_number = int(self.final_dataset['Label'].max()) + 1
                for new_cluster_element in new_cluster_elements.iterrows():
                    self.final_dataset.loc[
                        self.final_dataset.index[int(new_cluster_element[1])], 'Label'] = new_cluster_number

                print("A new cluster is now formed out of already existing outliers.")

                # The new cluster's mean core element is calculated after the cluster's creation.
                self.find_mean_core_element() 

            else:
                # The new element is an outlier.
                # It is not close enough to its closest in order to be added to it,
                # neither has enough outliers close by to form a new cluster.
                self.final_dataset.loc[self.final_dataset.index[-1], 'Label'] = -1

        print("The new element in the dataset: \n", self.final_dataset.tail(1))



    def find_largest_cluster(self):
        """
        This function identifies the largest of the clusters with respect to the number of the core elements.
        The largest cluster is the one with the most core elements in it.

        :returns: the number of the largest cluster. If -1 is returned, then there are no clusters created
        in the first place.
        """
        cluster_size = self.final_dataset.groupby('Label')['Label'].count() #算每个聚类簇的成员数量的方法
        # cluster_size = cluster_size['Temperature'].value_counts()
        #try:
        #    cluster_size = cluster_size.drop(labels=[-1])
        #except ValueError:
        #    print("The label -1 does not exist")
        cluster_size = cluster_size['Temperature'].value_counts()
        largest_cluster = -1 
        if not cluster_size.empty:
            largest_cluster = cluster_size.idxmax()
            print('The cluster with the most elements is cluster No: ', cluster_size.idxmax())
            return largest_cluster
        else:
            print('There aren\'t any clusters formed yet')
            return largest_cluster

    def find_cluster_limits(self):
        self.cluster_limits = self.final_dataset\
            .groupby(self.final_dataset['Label'])\
            .agg(['min', 'max'])
        print(self.cluster_limits)
        #self.cluster_limits.to_json(r'../json_exports/all_cluster_limits.json')

    def get_largest_cluster_limits(self):
        self.largest_cluster_limits = self.cluster_limits.iloc[self.largest_cluster+1]
        #self.largest_cluster_limits.to_json(r'../json_exports/largest_cluster_limits.json')
        print(self.largest_cluster_limits)

    def add_dataset(self,temperature,gas,relative_humidity):#加入一行
        self.dataset.loc[self.dataset.index[-1] + 1, 'Temperature'] = temperature
        self.dataset.loc[self.dataset.index[-1], 'Gas'] = gas
        self.dataset.loc[self.dataset.index[-1], 'Relative Humidity'] = relative_humidity


      
      
    def incremental_dbscan_(self): #IIII
        self.final_dataset = self.final_dataset.append({'Temperature': self.dataset.iloc[-1]['Temperature'],
                                                        'Gas': self.dataset.iloc[-1]['Gas'],
                                                        'Relative Humidity': self.dataset.iloc[-1]['Relative Humidity'],  #最后一行
                                                        'Label': -1}, ignore_index=True)  #则将忽略原始数据帧的索引，并为连接后的数据帧生成新的默认索引
        print(self.final_dataset.tail(n=1))
        self.find_mean_core_element()  #核心平均值，这一步感觉有点多余，不行除掉这个
        print(1)
        min_distance_mean_core_element_index = self.calculate_min_distance_centroid() #一个用于计算新元素与每个聚类簇的核心元素之间的距离，并找到距离最近的聚类簇的方法
        print(2)
        if min_distance_mean_core_element_index is not None:
            self.check_min_samples_in_eps_or_outlier(min_dist_index=min_distance_mean_core_element_index) #是一个用于检查距离最近的聚类簇是否满足最小样本数和半径阈值的条件
        print(3)
        self.largest_cluster = self.find_largest_cluster()
        print(4)
        self.find_cluster_limits()
        print(5)
        self.get_largest_cluster_limits()
        self.final_dataset.tail(n=1).to_csv('final_dataset.csv', mode='a', header=False, index=False)

    #def final_dataset_show_3d (self):

    def show_3d_init(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.data_points = {}
        self.colors = {}
        self.colors[-1] = 'r' 

        self.ax.set_xlabel('Temperature')
        self.ax.set_ylabel('Gas')
        self.ax.set_zlabel('Relative Humidity')

        self.ax.legend()
    
    def draw_3d (self):
        self.plot_thread.join()  
        for index, row in self.final_dataset.iterrows():
            temperature = row['Temperature']
            gas = row['Gas']
            relative_humidity = row['Relative Humidity']
            label = row['Label']
      
            for label in self.final_dataset['Label'].unique():
                 if label != -1:
                    self.colors[label] = plt.cm.jet(len(self.colors) / len(self.final_dataset['Label'].unique()))

            for label, row in self.final_dataset.iterrows():
                color = self.colors[row['Label']]
             # 绘制数据点
            self.ax.scatter(temperature, gas, relative_humidity, color=color, label=f'Label {label}')
            
        # 设置坐标轴标签
    
        self.plot_thread.start()
         # 显示图例



    def show_plot(self):
        self.plt.show()
