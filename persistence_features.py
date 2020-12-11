import gudhi as gd
from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import normalize
import numpy as np
import time

class PersistantHomology():
    def __init__(self,metric='euclidean',max_dimension=2,edge_threshold=0.8,M=5,N=5):
        self.metric = metric
        self.edge_threshold = edge_threshold
        self.max_dimension = max_dimension
        self.M = M
        self.N = N

    def calculate_distance_matrix(self,data_arr):
        measure = DistanceMetric.get_metric(self.metric)
        distance_matrix = measure.pairwise(normalize(data_arr))
        return distance_matrix

    def calculate_persistence(self,distance_matrix):
        skeleton = gd.RipsComplex(distance_matrix=distance_matrix,
                                  max_edge_length=self.edge_threshold)
        skeleton_tree = skeleton.create_simplex_tree(max_dimension=2)
        bar_codes = skeleton_tree.persistence()
        return bar_codes

    def calculate_features(self,bar_codes):
        h0 = []
        h1 = []
        for code in bar_codes:
            if code[0]==0:
                h0.append(code[1])
            elif code[0]==1:
                h1.append(code[1])

        h0,h1 = np.array(h0),np.array(h1)
        def remove_inf(arr,offset=1):
            inf_index = np.where(np.isinf(arr))
            if(inf_index[0].shape[0]!=0):
                arr[inf_index[0],inf_index[1]]=offset
            return arr
        h0,h1 = remove_inf(h0),remove_inf(h1)
        h = h0.T
        features = np.zeros((self.M,self.N))
        for m in range(self.M):
            for n in range(self.N):
                features[m][n] = sum(((h[0]+h[1])**m)*((h[1]-h[0])**n))
        retfeats = features
        features = np.zeros((self.M,self.N))
        h = h1.T
        if len(h1) != 0:
            for m in range(self.M):
                for n in range(self.N):
                    features[m][n] = sum(((h[0]+h[1])**m)*((h[1]-h[0])**n))
        retfeats = np.hstack((retfeats.flatten(),features.flatten())).flatten()

        return retfeats

    def generate_topological_feature(self,data_arr):
        dist_time = time.time()
        distance_matrix = self.calculate_distance_matrix(data_arr)
        dist_time = time.time() - dist_time

        bar_time = time.time()
        bar_codes = self.calculate_persistence(distance_matrix)
        bar_time = time.time() - bar_time

        feat_time = time.time()
        features = self.calculate_features(bar_codes)
        feat_time = time.time() - feat_time

        return features,dist_time,bar_time,feat_time
