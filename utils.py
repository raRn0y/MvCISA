import warnings
warnings.filterwarnings('ignore')
import scipy.io as sio
import numpy as np
import math
from numpy.random import shuffle
import torch
from typing import Union, List
from torch import cuda, normal
from sklearn.cluster import SpectralClustering, spectral_clustering
from sklearn.metrics import adjusted_rand_score
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import contingency_matrix
from scipy.special import comb
from sklearn import preprocessing
from sklearn.model_selection import KFold

def save_picture(traindata, testdata, whichTest):
    data = traindata.data['0'].reshape(-1, 64, 64) # attention to catch before '='
    map = np.concatenate(data)
    map = map.T*255
    import cv2
    cv2.imwrite(f'./result/train_map_after_kfold,test={whichTest}.jpg', map)

    data = testdata.data['0'].reshape(-1, 64, 64) # attention to catch before '='
    map = np.concatenate(data)
    map = map.T*255
    cv2.imwrite(f'./result/test_map_after_kfold,test={whichTest}.jpg', map)


class DataSet(object):

    def __init__(self, data, view_number, labels):
        """
        Construct a DataSet.
        """
        self.data_dict = dict() # dictionary for views
        self.view_cnt = view_number
        self._labels = labels
        for v_num in range(view_number):
            if view_number > 1:
                self.data_dict[v_num] = data[v_num]
                self._num_examples = data[0].shape[0] # the number of the examples in one view
            else:
                self.data_dict['0'] = data_dict

        self.data_list = []
        #for each in traindata.data.values():

        # force
        #view_number = 2
        cnt = 0
        for each in self.data_dict.values():
            self.data_list.append(each)
            cnt +=1
            if cnt == view_number:
                break



    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

def Normalize(data):
    """
    :param data:Input data
    :return:normalized data
    """
    m = np.mean(data)
    mx = np.max(data)
    mn = np.min(data)
    return ((data - m) / (mx - mn)).astype(np.float64)

def znormalize(data):
    return preprocessing.scale(data, axis=1)

def matlab_normalize(data:np.ndarray):
    data = data.T
    result = np.sqrt(np.sum(data ** 2, axis=0))
    result = np.repeat(result[None], data.shape[0], axis=0) + np.spacing(1)
    result = data / result
    result = result.T
    return result.astype(np.float64)

def read_matlab_data_clustering(mat_path:str, mat_name:str, Normal:int=1, bShuffle:bool=True, T = False):

    data = sio.loadmat(mat_path+mat_name)

    print(data.keys())
    datakeys = ['X', 'data']
    labelkeys = ['truth', 'Y', 'gt', 'truelabel', 'label', 'labels']
    key_data = None
    key_label = None

    for each in datakeys:
        if each in data.keys():
            key_data = each

    for each in labelkeys:
        if each in data.keys():
            key_label = each

    X = data[key_data]
    view_number = X.shape[1]
 
    print('view number:', end=' ')
    print(view_number)
    
    labels = data[key_label]
    #if len(labels) == 1:
        #labels = labels[0].tolist()
    
    
    if len(labels[0]) == 1:
        labels = labels.T[0]
        while len(labels) == 1:
            labels = labels[0]
        labels.dtype = np.uint8
    else:
        labels = labels[0]
    #print(labels)

    print('sample number:', end=' ')
    print(len(labels))

    if min(labels) == 0:
        labels = labels + 1
    else:
        labels = labels

    X = np.split(X, view_number, axis=1)#X is a list
    for i in range(len(X)):
        #if X[i].dtype == object:
        if False:
            tmp = X[i][0][0].toarray()
            #print(tmp.shape)
        else:
            if T:
                tmp = X[i][0][0].T
            else:
                tmp = X[i][0][0]

        a = [(tmp[x], labels[x]) for x in range(len(tmp))]
        a.sort(key=lambda x:x[1])
        labels = np.array([a[x][1] for x in range(len(tmp))])
        tmp = np.array([a[x][0].tolist() for x in range(len(tmp))])
        #tmp = np.array(tmp)
        #print(tmp.shape)
        X[i] = tmp

    if bShuffle:
        for i in range(view_number):
            np.random.seed(1)
            #print(X[i].shape)
            np.random.shuffle(X[i])
        np.random.seed(1)
        np.random.shuffle(labels)
        print('trainset is shuffled')
    print(labels)
    


    if Normal == 1:
        for v_num in range(view_number):
            X[v_num] = matlab_normalize(X[v_num])
    elif Normal == 2:
        for v_num in range(view_number):
            X[v_num] = Normalize(X[v_num])
    elif Normal == 3:
        for v_num in range(view_number):
            X[v_num] = znormalize(X[v_num])
    return X, labels, view_number

def read_matlab_data_kfold(mat_path:str, mat_name:str, K:int=0, whichTest:int=0, Normal:int=1, bShuffle:bool=True, which:int=1):
    """
    read data and spilt it train set and test set evenly
    ---
    :param mat_path:.mat filepath
    :param ratio:training set ratio
    :param Normal:do you want normalize
    ----
    :return:dataset and view number
    """

    data = sio.loadmat(mat_path+mat_name)

    print(data.keys())
    datakeys = ['X', 'data']
    labelkeys = ['truth', 'Y', 'gt', 'truelabel', 'label', 'labels']
    key_data = None
    key_label = None

    for each in datakeys:
        if each in data.keys():
            key_data = each

    for each in labelkeys:
        if each in data.keys():
            key_label = each


    if False:# data is [[[]], [[]]...]
        view_number = len(data[key_data])
        tmp = []
        for each in data[key_data]:
            tmp.append(np.array(each))
        X = tmp

    else:# data is [array, array, ...]
        X = data[key_data]
        view_number = X.shape[1]
        X = np.split(X, view_number, axis=1)#X is a list

    print('view number:', end=' ')
    print(view_number)
    
    X_train = []
    X_test = []
    labels_train = []
    labels_test = []

    labels = data[key_label]
    #if len(labels) == 1:
        #labels = labels[0].tolist()
    
    labels = labels.T[0]
    while len(labels) == 1:
        labels = labels[0]

    print('sample number:', end=' ')
    print(len(labels))

    if min(labels) == 0:
        labels = labels + 1
    else:
        labels = labels
    
    #print(labels) # from 1 to 15

    classes = int(np.max(labels))
    print('classes:', end=' ')
    print(classes)
  
    dic_X = {i:[] for i in range(1, classes+1)}


    # fold
    #print(labels.shape)
    #print(X[1][0][0].shape)
    for i in range(len(labels)):
        for view in range(view_number):
            if which == 10:# [view1, view2, ...] view=[[]]
                tmpX = X[view][0][0][i].toarray()[0]
                dic_X[labels[i]].append(tmpX)
            elif which == 12:
                tmpX = X[view][0][0].toarray().T[i]
                dic_X[labels[i]].append(tmpX)
            else:
                if which == 1 or which == 5 or which == 6 or which == 7 or which == 8 or which == 12:
                    tmpX = np.array(X[view][0][0]).T
                else:
            
                    tmpX = np.array(X[view][0][0])
                if i == 0:
                    print('shape of origin view:', end=' ')
                    print(tmpX.shape)
                    print('slicing at dimension 0, sample number: ', end=' ')
                    print(tmpX.shape[0])

                dic_X[labels[i]].append(tmpX[i].tolist())

    #print(len(dic_X[1][1][1]))
    dic_X_view = {i:[] for i in range(1, classes+1)}
    for i in range(1, classes+1):
        for view in range(view_number):
            dic_X_view[i].append(dic_X[i][view::view_number])

    dic_train = {view:[] for view in range(view_number)}
    dic_test = {view:[] for view in range(view_number)}
    #print(len(dic_X_view[1][1][1]))

    if K > 0:
        kf = KFold(n_splits=K)#shuffle=False
        for label in dic_X_view.keys():
            #print(dic_X_view[label][1])
            #for i in range(view_number):
            #print(torch.tensor(dic_X_view[label][i]).shape)
            #print(dic_X_view[label])
            #print(len(dic_X_view[label][0][0]))
            classSet = np.array(dic_X_view[label], dtype=object)
            labelSet = np.array([label for j in range(len(dic_X_view[label][0]))])
            cnt = 1
            for train_index, test_index in kf.split(labelSet):
                if cnt == whichTest:
                    #print(labelSet[train_index])
                    labels_train.extend(labelSet[train_index].tolist())
                    labels_test.extend(labelSet[test_index].tolist())
                    for view in range(view_number):
                        #print(classSet[view])
                        dic_train[view].extend(classSet[view][train_index])
                        dic_test[view].extend(classSet[view][test_index])
                    break
                cnt += 1

        #print(len(dic_train[0]))
        #print(len(labels_train))

        X_train = [np.array(Xv) for Xv in dic_train.values()]
        X_test = [np.array(Xv) for Xv in dic_test.values()]
         

        # save kfold data
        kfoldmatpath = './kfold-datasets/TrainSetFold'+str(whichTest)+'-of-'+mat_name
        print(kfoldmatpath)
        newdict = {}
        for each in dic_train.keys():
            newdict['view' + str(each)] = dic_train[each]
        newdict['labels'] = labels_train
        #sio.savemat(kfoldmatpath, newdict)

        kfoldmatpath = './kfold-datasets/TestSetFold'+str(whichTest)+'-of-'+mat_name
        newdict = {}
        for each in dic_train.keys():
            newdict['view' + str(each)] = dic_test[each]
        newdict['labels'] = labels_test
        #sio.savemat(kfoldmatpath, newdict)


        if bShuffle:
            for i in range(len(X_train)):
                np.random.seed(1)
                np.random.shuffle(X_train[i])
            np.random.seed(1)
            np.random.shuffle(labels_train)
            print('trainset is shuffled')
        
        '''
        tmp = []
        for ea in X_train:
            t=[]
            for each in ea:
                print(each)
                t.append(each.todense())
            tmp.append(tmp)
        X_train = np.array(tmp)
        #print(X_train)
        '''
        print("saving kfold-mat to path: ")

    else:
        #print(np.array(dic_X_view[1]).shape)
        X_train = []
            
        # unfold
        for i in range(1, classes+1):
            view = 0
            for each in dic_X[i]:
                X_train[view].append(each)
                view += 1
                if view == view_number:
                    view = 0
            part_l = [i for j in range(dic_l[i])]
            labels_train.extend(part_l)
        #print(X_train[0][0])
        for view in range(view_number):
            X_train[view] = np.array(X_train[view])
    #print(len(X_train[0]))

    # show dims
    
    for v_num in range(view_number):
        print('View '+str(v_num)+' for train:',end=' ')
        print(X_train[v_num].shape)
        print('View '+str(v_num)+' for test:',end='  ')
        print(X_test[v_num].shape)
            
    #print(labels)
    # normalize
    if Normal == 1:
        for v_num in range(view_number):
            # X_train[v_num] = Normalize(X_train[v_num])
            X_train[v_num] = matlab_normalize(X_train[v_num])
            if K > 0:
                # X_test[v_num] = Normalize(X_test[v_num])
                X_test[v_num] = matlab_normalize(X_test[v_num])
    elif Normal == 2:
        for v_num in range(view_number):
            # X_train[v_num] = Normalize(X_train[v_num])
            X_train[v_num] = Normalize(X_train[v_num])
            if K > 0:
                # X_test[v_num] = Normalize(X_test[v_num])
                X_test[v_num] = Normalize(X_test[v_num])
    elif Normal == 3:
        for v_num in range(view_number):
            # X_train[v_num] = Normalize(X_train[v_num])
            X_train[v_num] = znormalize(X_train[v_num])
            if K > 0:
                # X_test[v_num] = Normalize(X_test[v_num])
                X_test[v_num] = znormalize(X_test[v_num])
        
    if K > 0:
        traindata = DataSet(X_train, view_number, np.array(labels_train)-1)
        testdata = DataSet(X_test, view_number, np.array(labels_test)-1)
        return traindata, testdata, view_number
    else:
        traindata = DataSet(X_train, view_number, np.array(labels_train)-1)
        # testdata = DataSet(X_test, view_number, np.array(labels_test))
        return traindata, None, view_number

def hungarian(cost_mat:np.ndarray):
    """
    cost_mat: cost matrix of shape [num_class, num_cluster]

    ----
    return:
    matches: [num_class, 2] 分配矩阵
    """
    # matches = linear_assignment(cost_mat)
    _, matches = linear_sum_assignment(cost_mat)
    
    return matches

    

class Clustering(object):
    def __init__(self, cls_num:int, affinity='precomputed'):
        # self.affinity_mat = affinity_mat
        # self.labels = labels
        self.cls_num = cls_num
        #self.clustering_model = SpectralClustering(n_clusters=cls_num, affinity='precomputed')
        self.clustering_model = SpectralClustering(n_clusters=cls_num, affinity=affinity)

    def clustering(self, affinity_mat:Union[torch.Tensor, np.ndarray], labels):
        pred_labels = self.clustering_model.fit_predict(affinity_mat)
        '''
        print('pred-origin')
        print(pred_labels)
        #pred_labels = spectral_clustering(affinity=affinity_mat.numpy(), n_clusters=self.cls_num)
        '''
        best_matches_pred_labels = self.__best_map(labels, pred_labels)

        '''
        print('pred-bestmatch')
        print(best_matches_pred_labels)
        '''
        return best_matches_pred_labels

    
    def __best_map(self, label:np.ndarray, pred:np.ndarray):
        """
        采用匈牙利算法进行标签与预测簇的匹配
        """
        #print(label)
        #pred = pred+1
        #print(pred)
        assert len(label) == len(pred), 'predict and label not match'
        N = len(label)
        clusters = np.unique(pred)
        classes = np.unique(label)
        num_cluster = len(clusters)
        num_class = len(classes)
        G = np.zeros((num_class, num_cluster))
        for i in range(num_class):
            for j in range(num_cluster):
                G[i, j] = np.sum((label == classes[i]) * (pred == clusters[j]))

        # G[num_class, num_cluster]为cost矩阵
        # G[i, j]代表第i类的样本中属于第j个簇的数量
        # 利用匈牙利算法进行最大匹配, 需要用-G转换成最小匹配

        matches = hungarian(-G)
        #print(matches)
        new_pred = np.zeros_like(pred)
        for i in range(len(clusters)):
            #print(pred==clusters[i])
            # ex. i=5, clusters[5] = 0, classes
            #print('mat',matches[i])
            #print('right(class)',classes[matches[i]])
            #print('pred==',clusters[i])
            new_pred[pred == clusters[i]] = classes[matches[i]]

        return new_pred

    
    def calc_nmi(self, label:np.ndarray, pred:np.ndarray):
        N = len(label)
        clusters = np.unique(pred)
        classes = np.unique(label)
        num_cluster = len(clusters)
        num_class = len(classes)

        # compute number of points in each class
        cls_cnt = np.array(
            [np.sum(label == cls_id) for cls_id in classes]
        )
        cluster_cnt = np.array(
            [np.sum(pred == cluster_id) for cluster_id in clusters]
        )

        # mutual information
        mi = 0
        A = np.zeros((num_cluster, num_class))
        miarr = np.zeros_like(A)
        avgent = 0

        for i in range(num_cluster):
            index_cluster = pred == clusters[i]
            for j in range(num_class):
                index_class = label == classes[j]
                A[i, j] = np.sum(index_cluster * index_class)
                if A[i, j] != 0:
                    miarr[i, j] = A[i, j] / N * np.log2(N * A[i, j] / (cluster_cnt[i] * cls_cnt[j]))
                    avgent = avgent - (cluster_cnt[i] / N) * (A[i, j] / cluster_cnt[i]) * np.log2(A[i, j] / cluster_cnt[i])
                else:
                    miarr[i, j] = 0.0

                mi += miarr[i, j]
        
        # class_entropy
        cls_ent = np.sum(cls_cnt / N * np.log2(N / cls_cnt))

        # clustering entropy
        clust_ent = np.sum(cluster_cnt / N * np.log2(N / cluster_cnt))

        nmi = 2 * mi / (cls_ent + clust_ent)
        return nmi

    def cluster_acc(self, Y, Y_pred):
        from scipy.optimize import linear_sum_assignment as linear_assignment

        assert Y_pred.size == Y.size
        D = max(Y_pred.max(), Y.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(Y_pred.size):
            w[Y_pred[i], Y[i]] += 1
        ind = linear_assignment(w.max() - w)
        total = 0
        for i in range(len(ind[0])):
            total += w[ind[0][i], ind[1][i]]
        return total * 1.0 / Y_pred.size, w


    '''
        classes = torch.unique(Y)
        nClass = len(torch.unique(Y))
        G = torch.zeros(nClass)
        for i in range(nClass):
            for j in range(nClass):
                G[i][j] = len(Y==classes[i] & Y_pred==classes[j])
        c, t = cugraph.linear_assignment.hungarian(-G)
        newPred = torch.zeros(Y_pred.shape)
        for i in range(nClass):
            newPred[Y_pred==Class[i]] = Y[c[i]]
    '''

    def calc_acc(self, label:np.ndarray, pred:np.ndarray):
        #print(label == pred)
        #print(label.tolist())
        #print(pred.tolist())
        if isinstance(label, torch.Tensor):
            acc = np.sum(label == pred) / len(label)
        else:
            acc = np.sum(label == pred) / len(label)
        return acc
    
    def calc_f(self, label:np.ndarray, pred:np.ndarray):
        assert len(label) == len(pred), 'predict and label not match'
        N = len(label)
        num_true, num_pred, num_i = 0, 0, 0
        for i in range(N):
            label_n = label[i + 1:] == label[i]
            pred_n = pred[i + 1:] == pred[i]
            num_true += np.sum(label_n)
            num_pred += np.sum(pred_n)
            num_i += np.sum(label_n * pred_n)
        
        p, r, f = 1, 1, 1
        if num_pred > 0:
            p = num_i / num_pred
        if num_true > 0:
            r = num_i / num_true
        if p + r == 0:
            f = 0
        else:
            f = 2 * p * r / (p + r)
        
        return f, p, r
  

    def calc_rand_index(self, label:np.ndarray, pred:np.ndarray, adjusted:bool=True):
        if adjusted:
            return adjusted_rand_score(label, pred)
        else:
            assert len(label) == len(pred), 'predict and label not match'
            N = len(label)
            C = contingency_matrix(label, pred) # contingency matrix
            n = np.sum(C)
            nis = np.sum(np.sum(C, axis=1) ** 2) # sum of squares of sums of rows
            njs = np.sum(np.sum(C, axis=0) ** 2) # sum of squares of sums of columns

            t1 = comb(n, 2) # total number of pairs of entities
            t2 = np.sum(C ** 2) # sum over rows & columnns of nij^2
            t3 = 0.5 * (nis + njs)

            # Expected index (for adjustment)
            nc = (n * (n ** 2 + 1) - (n + 1) * (nis + njs)+ 2 * (nis * njs) / n) / \
                (2 * (n - 1))
            A = t1 + t2 - t3 # no. agreements
            D = t3 - t2 # no. disagreements
            if t1 == nc:
                AR = 0
            else:
                AR = (A - nc) / (t1 - nc) # adjusted Rand - Hubert & Arabie 1985
            
            RI = A / t1 # Rand 1971 %Probability of agreement
            MI = D / t1 # Mirkin 1970 %p(disagreement)
            HI = (A - D) / t1 # Hubert 1977 %p(agree)-p(disagree)
            return AR, RI, MI, HI

def RI(label:np.ndarray, pred:np.ndarray, adjusted:bool=True):
    assert len(label) == len(pred), 'predict and label not match'
    N = len(label)
    C = contingency_matrix(label, pred) # contingency matrix
    n = np.sum(C)
    nis = np.sum(np.sum(C, axis=1) ** 2) # sum of squares of sums of rows
    njs = np.sum(np.sum(C, axis=0) ** 2) # sum of squares of sums of columns

    t1 = comb(n, 2) # total number of pairs of entities
    t2 = np.sum(C ** 2) # sum over rows & columnns of nij^2
    t3 = 0.5 * (nis + njs)

    # Expected index (for adjustment)
    nc = (n * (n ** 2 + 1) - (n + 1) * (nis + njs)+ 2 * (nis * njs) / n) / \
        (2 * (n - 1))
    A = t1 + t2 - t3 # no. agreements
    D = t3 - t2 # no. disagreements
    if t1 == nc:
        AR = 0
    else:
        AR = (A - nc) / (t1 - nc) # adjusted Rand - Hubert & Arabie 1985
            
    RI = A / t1 # Rand 1971 %Probability of agreement
    MI = D / t1 # Mirkin 1970 %p(disagreement)
    HI = (A - D) / t1 # Hubert 1977 %p(agree)-p(disagree)
    return AR, RI, MI, HI

if __name__ == "__main__":
    label = np.array([0, 1, 0, 1, 2])
    pred = np.array([0, 1, 0, 0, 2])
    print(RI(label, pred))
            

    
