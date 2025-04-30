import scipy.io as sio  
import matplotlib.pyplot as plt  
import numpy as np  
import scipy 
import scipy.io as sio
import _pickle as pk
import argparse
import torch
from torch import optim
import utils
from tqdm import tqdm
import os
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import math
import cv2
from itertools import permutations, combinations
from mcl_losses import SupMCL


#计算需要几个特征值
def eigValPct(eigVals,percentage):
    sortArray=np.sort(eigVals) 
    sortArray=sortArray[-1::-1] 
    arraySum=sum(sortArray) 
    tempSum=0
    num=0
    for i in sortArray:
        tempSum+=i
        num+=1
        if tempSum>=arraySum*percentage:
            return num
            
#计算保留percentage百分比信息量的pca
def pca(dataMat,percentage=0.9):
    meanVals=np.mean(dataMat,axis=0)  
    meanRemoved=dataMat-meanVals
    covMat=np.cov(meanRemoved,rowvar=0)  
    eigVals,eigVects=np.linalg.eig(np.mat(covMat)) 
    k=eigValPct(eigVals,percentage) 
    eigValInd=np.argsort(eigVals)  
    eigValInd=eigValInd[:-(k+1):-1]
    redEigVects=eigVects[:,eigValInd]  
    lowDDataMat=meanRemoved*redEigVects 
    return lowDDataMat
    
#初始化参数W
def xavier_init(n_in, n_out, constant = 1): #constant = 4
    low = - constant * np.sqrt(6.0 / (n_in + n_out))
    high = constant * np.sqrt(6.0 / (n_in + n_out))
    return np.random.uniform(low, high, size=[n_in,n_out])
    
#设置参数V
def pro_v(n_in, n_out, num = 2):  # n_in / n_out must equal  num, and n_in%n_out ==0
    v = np.zeros([n_in, n_out])
    for i in range(n_out):
         for j in range(num):
            v[i*num+j][i] = 1
    return v
    
def deal_nan(M):
    inf = M.max()
    if isinstance(M, torch.Tensor):
        if torch.isnan(M).any():#True
            print('WARNING: ISA has nan')
            M = torch.where(torch.isnan(M), torch.full_like(M, 0.), M)
        if torch.isinf(M).any():#True
            print('WARNING: ISA has inf')
            M = torch.where(torch.isinf(M), torch.full_like(M, inf), M)
        #print(M)

    else:
        where_are_nan = np.isnan(M)
        print(where_are_inf.any())
        where_are_inf = np.isinf(M)
        print(where_are_inf.any())
        M[where_are_nan] = 0
        M[where_are_inf] = inf
    return M
def Normalize(data):
    """
    :param data:Input data
    :return:normalized data
    """
    m = np.mean(data)
    mx = np.max(data)
    mn = np.min(data)
    return ((data - mn) / (mx - mn)).astype(np.float64)
    #return (data+mn).astype(np.float64)
def Mabs(M):
    if isinstance(M, torch.Tensor):
        abs=torch.maximum(M,-M)
    else:
        abs=np.maximum(M,-M)
    return abs

    
#制作4个ISASP网络的输入数据   
def pro_data(fea_, image_size = 32, patch_size = 20):
    #print("fea_.shape")
    #print(fea_.shape)
    fea1 = []
    fea2 = []
    fea3 = []
    fea4 = []
    for item in fea_:
        # 0~10, 0~10
        # 6~16, 6~16
        # 0~10, 6~16
        # 6~16, 0~10
        tmp1 = item.reshape(image_size,image_size)[0:patch_size,0:patch_size].reshape(patch_size*patch_size)
        fea1.append(tmp1)
        tmp2 = item.reshape(image_size,image_size)[image_size-patch_size:image_size,image_size-patch_size:image_size].reshape(patch_size*patch_size)
        fea2.append(tmp2)
        tmp3 = item.reshape(image_size,image_size)[0:patch_size,image_size-patch_size:image_size].reshape(patch_size*patch_size)
        fea3.append(tmp3)
        tmp4 = item.reshape(image_size,image_size)[image_size-patch_size:image_size,0:patch_size].reshape(patch_size*patch_size)
        fea4.append(tmp4)
    fea1 = np.array(fea1)
    fea2 = np.array(fea2)
    fea3 = np.array(fea3)
    fea4 = np.array(fea4)
    return fea1, fea2 ,fea3, fea4

def fractional_matrix_power(m, frac = (-1/2)):
    if torch.isinf(m).any():
        print('ERROR: m has inf ')
        print(m)
    elif torch.isnan(m).any():
        print('ERROR: m has nan:')
        print(m)
    evals, evecs = torch.linalg.eig (m)#, eigenvectors = True)  # get eigendecomposition
    evals = torch.real(evals)                                # get real part of (real) eigenval
    evpow = (0.00001+evals)**(frac)                              # raise eigenvalues to fractional power
    if torch.isnan(evpow).any():#True
        print('WARNING: eval**fraction has nan')
        evpow = torch.where(torch.isnan(evpow), torch.full_like(evpow, 0.), evpow)
    if torch.isinf(evpow).any():#True
        print('WARNING: eval**fraction has inf')
        evpow = torch.where(torch.isinf(evpow), torch.full_like(evpow, 1.0), evpow)
        #print(evpow)


    # build exponentiated matrix from exponentiated eigenvalues
    a = torch.diag (evpow).to(torch.float64)
    b = torch.inverse (evecs).to(torch.float64)
    mpow = torch.matmul (evecs.to(torch.float64), torch.matmul (a, b).to(torch.float64))
    return mpow


def cluster(X, gnd, affinity='rbf', path = None):
    cluster_model = utils.Clustering(len(np.unique(gnd)), affinity)
    pred_labels = cluster_model.clustering(X, gnd)

    acc = cluster_model.cluster_acc(gnd, pred_labels)
    print("ACC1 =", str(acc[0]), end='\t')
    #acc = cluster_model.calc_acc(gnd, pred_labels)
    #print("ACC2 =", str(acc))
    from sklearn.metrics import normalized_mutual_info_score
    nmi = normalized_mutual_info_score(gnd, pred_labels)
    print("NMI =", nmi)
    if not path == None:
        f = open(path, 'w+')

        print('gnd')
        print(gnd)
        f.write('real_labels = [')
        for each in gnd:
            f.write(str(each) + ', ')
        f.write(']\n')

        print('pred')
        print(pred_labels)
        f.write('pred_labels = [')
        for each in pred_labels:
            f.write(str(each) + ', ')
        f.write(']\n')

        f.write(str(acc[0]))
        f.close()



class ISA_block():
    def __init__(self, Xshape, n_mid, n_out, num_of_sum, p=0.7, W=None, cuda_flag=True):
        self.n_sample, self.n_in = Xshape
        self.n_mid = n_mid
        self.n_out = n_out
        self.p = p
        #self.batch_size = batch_size

        if W is None:
            self.W = xavier_init(self.n_in, self.n_mid)
        else:
            self.W = W
        #print('applying constraint W*W.T=E...')
        #self.W_ = np.dot(self.W.T, self.W)
        #self.W = np.dot(W, np.real(scipy.linalg.fractional_matrix_power(W_, -1/2)))
        self.V = pro_v(n_mid, n_out, num_of_sum)
        self.Y = np.random.uniform(0, 1, size=[self.n_sample, self.n_out])
        self.cuda_flag = cuda_flag
        self.tensor_matrix()
        #self.orthogonal_constraint()
        print('W.shape')
        print(self.W.shape)
        print('V.shape')
        print(self.V.shape)

        #self.projection = torch.nn.Sequential(
        #    torch.nn.Linear(self.n_out, 160),
        #    torch.nn.ReLU(),
        #    torch.nn.Linear(160, 80)
        #).double()
        #if self.cuda_flag:
        #    self.projection.cuda()

    
    def tensor_matrix(self):
        self.W = torch.tensor(self.W)
        if self.cuda_flag:
            self.W = self.W.cuda()
        self.W.requires_grad_(True)
        #self.W.retain_grad()
        #print(self.W.is_leaf)
        self.V = torch.tensor(self.V)
        if self.cuda_flag:
            self.V = self.V.cuda()
        #self.V.requires_grad_(True) # don't !

    def orthogonal_constraint(self):
        print('applying constraint W*W.T=E...')
        # W正交矩阵
        with torch.no_grad():
            self.W_ = torch.mm(self.W.T, self.W)
            self.W = torch.mm(self.W, torch.real(fractional_matrix_power(self.W_, -1/2)))
            #self.W = deal_nan(self.W)
        self.W.requires_grad_(True)
            
    
    def forward(self, X):
        Y_ISA = self.ISA(X)
        #embedding = self.projection(Y_ISA)
        return Y_ISA#, embedding

    #定义ISA
    def ISA(self, X): # return yt
        epsilon=0.000001
        p = self.p
       
        if isinstance(X, torch.Tensor):
            if p > 1:                
                Y = (epsilon + torch.mm((torch.mm(X, self.W)) **p, self.V))**(1-(1/p))
            else:
                Y = (epsilon + torch.mm(Mabs(torch.mm(X, self.W)) ** p, self.V))**2#(1-(1/p))
        else:
            if p > 1:
                Y = (epsilon + np.dot((np.dot(X, self.W)) **p , self.V))**(1-(1/p))
            else:
                Y = (epsilon + np.dot(Mabs(np.dot(X, self.W)) ** p, self.V))**(1-(1/p))

        #Y = deal_nan(Y)
        return Y

    def reconstruction_loss(self, X):
        #print(X.shape)
        #print(self.W.shape)
        return torch.norm(torch.mm(X, torch.mm(self.W, self.W.T))-X)


   
    
#class ISA_clustering(torch.nn.Module):
class ISA_clustering():
    def __init__(self, view_blocks, blocks, n_out, max_iter , lrY, lrR, 
                 epochsR, epochsY, epochsC, epochsL, cuda_flag = True, 
                 batch_size = None, traindata = None, testdata = None, path = None):

        self.cuda_flag = cuda_flag
        self.view_blocks = view_blocks
        self.blocks = blocks

        self.max_iter = max_iter
        self.lrY = lrY
        self.lrR = lrR
        self.weight_decay = 0#1e-3
        self.epochsR = epochsR 
        self.epochsY = epochsY
        self.epochsC = epochsC
        self.epochsL = epochsL
        self.n_sample = traindata.data_list[0].shape[0]
        self.batch_size = batch_size
        self.n_batch = self.n_sample//self.batch_size
        self.traindata = traindata
        self.testdata = testdata
        self.path = path
        print('n_batch')
        print(self.n_batch)
        print('batch_size')
        print(self.batch_size)

        self.view_xs = []
        for each in traindata.data_list:
            if self.cuda_flag:
                self.view_xs.append(torch.tensor(each).cuda())
            else:
                self.view_xs.append(torch.tensor(each))

        self.test_xs = []
        for each in self.testdata.data_list:
            if self.cuda_flag:
                self.test_xs.append(torch.tensor(each).cuda())
            else:
                self.test_xs.append(torch.tensor(each))

        if self.cuda_flag:
            for b in self.view_blocks+blocks:
                b.W = b.W.cuda()
                b.V = b.V.cuda()

        self.gnd = torch.tensor(traindata.labels).type(torch.int64)
        if self.cuda_flag:
            self.gnd = self.gnd.cuda()

        self.testgnd = torch.tensor(self.testdata.labels).type(torch.int64)
        if self.cuda_flag:
            self.testgnd = self.testgnd.cuda()
        self.device = self.gnd.device


    def tensor_matrix(self):
        for each in self.view_blocks + self.blocks:
            each.tensor_matrix() # retransform W into leaf node
        # this W not that W
    def set_leaf(self):
        for b in self.view_blocks:
            b.W = b.W.detach()
            b.W.requires_grad = True
            #print(b.W.is_leaf) # True
            #print(b.W.T.is_leaf) # False
             

    def evaluate(self, output, gnd):
        _, predicted = torch.max(output, 1)
        if self.cuda_flag:
            predicted = np.array(predicted.cpu())
        else:
            predicted = np.array(predicted)
        gnd = np.array(gnd.cpu())
        correct = (predicted == gnd).sum().item()
        acc = 100*correct / len(gnd)
        #print('predicted:')
        #print(predicted)
        #print('gnd:')
        #print(gnd)
        #print('acc: '+str(acc))
        #output = torch.nn.functional.softmax(output)#.squeeze(1)
        #print(output.shape)
        #print(torch.tensor(gnd).shape)
        
        #loss = criterion(output, torch.tensor(gnd))
        return acc, predicted


    def InfoNCE_loss(self, q, k, op=1, T=False):
        q = torch.nn.functional.normalize(q, dim=1) # introduced from M3DM, if is necessary?
        k = torch.nn.functional.normalize(k, dim=1) # introduced from M3DM, if is necessary?
        #print(k.shape) # sample_number,  node_number
        if T:
            logits = torch.einsum('cn,cm->nm', [q, k])
        else:
            logits = torch.einsum('nc,mc->nm', [q, k])
        N = logits.shape[0]
        if op == 1:
            labels = torch.arange(N)
        elif op == 0:
            labels = torch.zeros(N, dtype=torch.long)
        if self.cuda_flag:
            labels = labels.cuda()
        return torch.nn.CrossEntropyLoss()(logits, labels)

    def CFO_loss(self, loss_type = 'info'):
        # 'cos': space contrast with cosine similarity

        cos_simi = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

        loss_cfo = torch.tensor(0, device = self.device)

        #ws = torch.cat([b.W for b in self.view_blocks])
        #for index, each in enumerate(self.view
        for b in self.view_blocks:
            #print('mul')
            ws = b.W
            subspaces = []
            num_subspaces = int(self.view_blocks[0].W.shape[1]/2)
            #print(num_subspaces)
            for i in range(num_subspaces):
                #print(ws[:,i*2].shape)
                subspace = torch.multiply(ws[:,i*2], ws[:,i*2+1])
                subspaces.append(subspace.unsqueeze(0))

            if loss_type == 'cos':
                comb = combinations(subspaces, 2)
                #print('cal cos')
                for c in comb:
                    loss_cfo = loss_cfo + cos_simi(c[0], c[1])
            elif loss_type == 'info':
                subspaces = torch.concat(subspaces, dim=0)
                #print(subspaces.shape)
                loss_cfo = self.InfoNCE_loss(subspaces, subspaces.detach().clone(), op=0)
            #print(loss_cfo)
        return loss_cfo







    def feature_extraction(self, view_xs, eval_flag = False, w_flag = False, cfo_flag = False):
        loss_H = torch.tensor(0, device = self.device)
        loss_W_InfoNCE = torch.tensor(0, device = self.device)
        view_ys = []
        #embeddings = []
        
        for v in range(len(self.view_blocks)):
            block = self.view_blocks[v]
            #print('q.grad')
            #print(q.grad)
            #print('vbv.w.grad')
            #print(self.view_blocks[v].W.grad)
            #jq.requires_grad = True
            #k = torch.zeros_like(block.W.T)
            Y_ISA = block.forward(view_xs[v])
            view_ys.append(Y_ISA)
            #embeddings.append(embedding)
           
            if w_flag:
                q = block.W#.detach()
                k = block.W.clone().detach()#.detach()#.clone().detach()
                #!loss_W_InfoNCE = loss_W_InfoNCE + self.InfoNCE_loss(q, k, 0, T=True)
                loss_W_InfoNCE = loss_W_InfoNCE + self.InfoNCE_loss(q, k, 1, T=True)
                #print(self.view_blocks[0].W.grad)
                #print(self.view_blocks[0].W.is_leaf)
            if eval_flag:
                loss_H = loss_H + Y_ISA.sum()

        if cfo_flag:
            loss_W_InfoNCE = self.CFO_loss()

        return view_ys, loss_H, loss_W_InfoNCE


    def model_forward(self, view_xs, gnd, eval_flag = True, cfo_flag = False):
        # everything everywhere all at once
        loss_C = torch.tensor(0, device = self.device)
        loss_H = torch.tensor(0, device = self.device)
        loss_W_InfoNCE = torch.tensor(0, device = self.device)
        loss_F_InfoNCE = torch.tensor(0, device = self.device)


        view_ys, loss_H, loss_W_InfoNCE = self.feature_extraction(view_xs, eval_flag = eval_flag, cfo_flag = cfo_flag)

        # Fusion Block
        fea = torch.cat(view_ys, 1).to(torch.float64)
        
        # Classification Block
        block = self.blocks[0]
        Y_ISA = block.ISA(fea)
        if eval_flag:
            '''
            loss_F_InfoNCE = torch.tensor(0)
            for element in permutations(view_ys, 2):
                loss_F_InfoNCE = loss_F_InfoNCE + self.InfoNCE_loss(element[0], element[1], 1)
            '''
            criterion_mcl = SupMCL(len(view_ys), view_ys[0].shape[1], 0, 0.1)
            #loss_vcl, loss_soft_vcl, loss_icl, loss_soft_icl = criterion_mcl(view_ys, gnd)
            loss_icl = criterion_mcl.icl(view_ys, gnd)
            loss_F_InfoNCE = loss_icl
            loss_Y = Y_ISA.sum()
            loss_H = loss_H + loss_Y
            criterion = torch.nn.CrossEntropyLoss()
            loss_C = criterion(Y_ISA, gnd)

        return Y_ISA, loss_C, loss_H, loss_W_InfoNCE, loss_F_InfoNCE

    def model_orthogonal_constraint(self):
        for b in self.view_blocks + self.blocks:
            b.orthogonal_constraint()

    def model_othogonal_loss(self):
        loss = torch.tensor(0, device = self.device)
        for b in self.view_blocks+self.blocks:
            W = b.W
            #print(torch.mm(W.T, W).shape)
            loss_O = torch.norm(torch.mm(W.T, W) - torch.eye(W.shape[1], device = self.device))
            loss = loss + loss_O

        return loss






    def testdata_evaluate(self):
        output = self.model_forward(self.test_xs, self.testgnd, eval_flag = False)[0]
        acc, pred = self.evaluate(output, torch.tensor(self.testdata.labels))
        #eval(output, self.testdata.labels)
        return acc, pred

    def save_feature(self):
        for v in range(len(self.view_blocks)):
            fea = self.view_blocks[v].ISA(self.view_xs[v]).cpu().detach().numpy()

            sio.savemat(self.path + 'fea-view'+str(v)+'.mat', {'fea':fea, 'gnd':self.gnd.cpu().numpy()})
        #sio.savemat(self.path + 'fea-fusion.mat', {'fea':self.blocks[0].ISA(self.view_xs[0]).detach().numpy()})

        #for i in range(len(fea0)):
        #    
        #    fea0i = Normalize(fea0[i])*255
        #    fea1i = Normalize(fea1[i])*255
        #    row_num = int((self.view_blocks[0].n_out)**0.5)

        #    cv2.imwrite(self.path + 'fea-view0-'+str(i)+'.jpg', fea0i.reshape(row_num, -1))
        #    cv2.imwrite(self.path + 'fea-view1-'+str(i)+'.jpg', fea1i.reshape(row_num, -1))
        #    #cv2.imwrite(self.path + 'fea-fusion.jpg', self.blocks[0].ISA(self.view_xs[0]).view(10, 20))

    def save_W(self):
        W0 = self.view_blocks[0].W.cpu().detach().numpy().T
        W1 = self.view_blocks[1].W.cpu().detach().numpy().T
        print(W0.shape)

        sio.savemat(self.path + 'W-view0.mat', {'W':W0})
        #sio.savemat(self.path + 'W-fusion.mat', {'W':self.blocks[0].ISA(self.view_xs[0]).detach().numpy()})

        for i in range(len(W0)):
            
            W0i = Normalize(W0[i])*255
            W1i = Normalize(W1[i])*255

            row_num = int((W0.shape[1])**0.5)

            try:
                cv2.imwrite(self.path + 'W-view0-'+str(i)+'.jpg', W0i.reshape(row_num, -1))
            except:
                print('failed to save W')
            #cv2.imwrite(self.path + 'W-fusion.jpg', self.blocks[0].ISA(self.view_xs[0]).view(10, 20))

    def R_optimize(self):
        optimR = optim.Adam(             
            [b.W for b in self.blocks+self.view_blocks],# + [p.data for b in self.blocks for p in b.projection.parameters()],
            lr=self.lrR,
            amsgrad=True, 
            weight_decay=self.lrR#weight_decay # this parameter in adam strongly influences the result!
        )

        scheduleR = CosineAnnealingWarmRestarts(optimR, T_0=100, T_mult=2, eta_min=self.lrR*0.01)
        bar =  tqdm(range(self.epochsR), leave = True)
        e = 0
        losses = []
        trainaccs = []
        testaccs = []
        for epoch in bar:
            output, loss_C, loss_Y, loss_W_InfoNCE, loss_F_InfoNCE = self.model_forward(self.view_xs, self.gnd)
            loss = loss_C
            acc, pred = self.evaluate(output, self.gnd)
            trainaccs.append(acc)
            #print('trainacc: ' + str(acc))
            optimR.zero_grad()
            loss.backward()
            optimR.step()
            testacc, testpred = self.testdata_evaluate()
            testaccs.append(testacc)
            #print('testacc:  ' + str(testacc))
            bar.set_description('                         loss_C: %.4f, F_loss: %.4f, process' % (loss_C.detach().item(), 0.1*loss_F_InfoNCE.detach().item()))
            losses.append(loss_C.detach().item())
            e+=1

            file = open(self.path+'pred-'+str(self.epochL)+'_'+str(epoch+1)+'.txt', 'w+')
            file.write('%.3f-%.3f   %.3f-%.3f\n' % (np.array(trainaccs).mean(), np.array(testaccs).mean(), np.array(trainaccs).max(), np.array(testaccs).max()))
            file.write('pred_labels:\n[')
            #file.write(str(pred))
            for i in testpred:
                file.write(str(i) + ', ')
            file.write(']\n')
            file.write('real_labels:\n[')
            for i in self.testdata.labels:
                file.write(str(i) + ', ')
            file.write(']\n')
            file.close()

            if e == 100:
                print(str(self.epochL)+'_'+str(epoch+1))
                print('trainacc best: %.3f' % np.array(trainaccs).max())
                print('trainacc avg: %.3f' % np.array(trainaccs).mean())
                print('trainacc min: %.3f' % np.array(trainaccs).min())
                print('testacc best: %.3f' % np.array(testaccs).max())
                print('testacc avg: %.3f' % np.array(testaccs).mean())
                print('testacc min: %.3f' % np.array(testaccs).min())
                print('formatted: %.3f-%.3f   %.3f-%.3f' % (np.array(trainaccs).mean(), np.array(testaccs).mean(), np.array(trainaccs).max(), np.array(testaccs).max()))
                
                scheduleR.step()
                e=0
                losses = []
                trainaccs = []
                testaccs = []
     

    def IR_optimize(self):
        optimR = optim.Adam(             
            [b.W for b in self.blocks+self.view_blocks],# + [p.data for b in self.blocks for p in b.projection.parameters()],
            lr=self.lrR,
            amsgrad=True, 
            #weight_decay=1e-3#weight_decay # this parameter in adam strongly influences the result!
            weight_decay=self.lrR#weight_decay # this parameter in adam strongly influences the result!
        )

        scheduleR = CosineAnnealingWarmRestarts(optimR, T_0=100, T_mult=2, eta_min=self.lrR*0.01)
        bar =  tqdm(range(self.epochsR), leave = True)
        e = 0
        losses = []
        trainaccs = []
        testaccs = []
        for epoch in bar:
            #for b in self.view_blocks+self.blocks:
            #    if torch.isnan(b.W).any():
            #        print('has nan000000000000000000000000000000000000000000000')
            #    else:
            #        print(b.W)
            output, loss_C, loss_Y, loss_W_InfoNCE, loss_F_InfoNCE = self.model_forward(self.view_xs, self.gnd, cfo_flag=True)
            loss = loss_C + 0.1 * loss_F_InfoNCE #+ 0.1 * loss_W_InfoNCE
            acc, pred = self.evaluate(output, self.gnd)
            trainaccs.append(acc)
            optimR.zero_grad()
            loss.backward()
            optimR.step()
            #print('trainacc: ' + str(acc))
            testacc, testpred = self.testdata_evaluate()
            testaccs.append(testacc)
            #print('testacc:  ' + str(testacc))
            bar.set_description('                         loss_C: %.4f, F_loss: %.4f, process' % (loss_C.detach().item(), 0.1*loss_F_InfoNCE.detach().item()))
            losses.append(loss_C.detach().item())
            e+=1

            file = open(self.path+'pred-'+str(self.epochL)+'_'+str(epoch+1)+'.txt', 'w+')
            file.write('%.3f-%.3f   %.3f-%.3f\n' % (np.array(trainaccs).mean(), np.array(testaccs).mean(), np.array(trainaccs).max(), np.array(testaccs).max()))
            file.write('pred_labels:\n[')
            #file.write(str(pred))
            for i in testpred:
                file.write(str(i) + ', ')
            file.write(']\n')
            file.write('real_labels:\n[')
            for i in self.testdata.labels:
                file.write(str(i) + ', ')
            file.write(']\n')
            file.close()

            if e == 10:
                print(str(self.epochL)+'_'+str(epoch+1))
                print('trainacc best: %.3f' % np.array(trainaccs).max())
                print('trainacc avg: %.3f' % np.array(trainaccs).mean())
                print('trainacc min: %.3f' % np.array(trainaccs).min())
                print('testacc best: %.3f' % np.array(testaccs).max())
                print('testacc avg: %.3f' % np.array(testaccs).mean())
                print('testacc min: %.3f' % np.array(testaccs).min())
                print('formatted: %.3f-%.3f   %.3f-%.3f' % (np.array(trainaccs).mean(), np.array(testaccs).mean(), np.array(trainaccs).max(), np.array(testaccs).max()))
                
                scheduleR.step()
                e=0
                losses = []
                trainaccs = []
                testaccs = []
        return testaccs
                

    def Y_optimize(self, soft_flag = False):
        optimY = optim.SGD(             
            [block.W for block in self.view_blocks+self.blocks],
            lr=self.lrY,
            #amsgrad=True, 
            #weight_decay=weight_decay # this parameter in adam strongly influences the result!
        )

        scheduleY = CosineAnnealingWarmRestarts(optimY, T_0=100, T_mult=2, eta_min=self.lrY*0.01)
    
        bar =  tqdm(range(self.epochsY), leave = True)

        for j in bar:
            loss = torch.tensor(0, device = self.device)
            output, loss_C, loss_Y, loss_W_InfoNCE, loss_F_InfoNCE = self.model_forward(self.view_xs, self.gnd)
            optimY.zero_grad()
            loss_O = self.model_othogonal_loss()
            if soft_flag:
                loss = loss_Y + 1000*loss_O
                loss.backward()
                optimY.step()
            else:
                loss = loss_Y
                loss.backward()
                optimY.step()
                self.model_orthogonal_constraint()
            bar.set_description('loss_Y: %.3f, loss_O: %.3f, process' % (loss_Y.detach().item(), loss_O.detach().item()))
            scheduleY.step()

        print('dataset classification result')
        acc, pred = self.evaluate(output, self.gnd)
        print(acc)
        #cluster(Y.detach(), self.gnd)

        self.save_feature()
        self.save_W()
    
       
    def VCL_optimize(self):
        self.set_leaf()

        # copy model
        self.Ws = [ori.W for ori in self.view_blocks]

        optimY = optim.SGD(             
            [block.W for block in self.view_blocks],
            lr=10,
            #amsgrad=True, 
            #weight_decay=weight_decay # this parameter in adam strongly influences the result!
        )

        #scheduleY = CosineAnnealingWarmRestarts(optimY, T_0=100, T_mult=2, eta_min=self.lrY*0.01)
    
        #bar =  tqdm(range(self.epochsR), leave = True)
        bar =  tqdm(range(self.epochsC), leave = True)

        for j in bar:
            output, loss_C, loss_Y, loss_W_InfoNCE, loss_F_InfoNCE = self.model_forward(self.view_xs, self.gnd, eval_flag = False, cfo_flag = True)
            optimY.zero_grad()
            loss_W_InfoNCE.backward()
            #print(self.view_blocks[0].W.grad)
            optimY.step()

            bar.set_description('loss_W_InfoNCE: %.3f, process' % loss_W_InfoNCE.detach().sum())
            #scheduleY.step()

        self.fusion_feature = output
        #for i in range(len(self.view_blocks)):
        #    self.view_blocks[i].W = Ws[i]
        #cluster(Y.detach(), self.gnd)

    def ICL_optimize(self):
        # currently works for 2 view dataset only
        self.set_leaf()

        # copy model
        self.Ws = [ori.W for ori in self.view_blocks]

        optimY = optim.SGD(             
            [block.W for block in self.view_blocks],
            #lr=self.lrY,
            lr = 0.1
            #amsgrad=True, 
            #weight_decay=weight_decay # this parameter in adam strongly influences the result!
        )

        #scheduleY = CosineAnnealingWarmRestarts(optimY, T_0=100, T_mult=2, eta_min=self.lrY*0.01)
    
        bar =  tqdm(range(self.epochsC), leave = True)

        for j in bar:
            output, loss_C, loss_Y, loss_W_InfoNCE, loss_F_InfoNCE = self.model_forward(self.view_xs, self.gnd)
            optimY.zero_grad()
            loss_F_InfoNCE.backward()
            #print(self.view_blocks[0].W.grad)
            optimY.step()

            bar.set_description('loss_F_InfoNCE: %.3f, process' % loss_F_InfoNCE.detach().sum())
            #scheduleY.step()

        self.fusion_feature = output
        #for i in range(len(self.view_blocks)):
        #    self.view_blocks[i].W = Ws[i]
        #cluster(Y.detach(), self.gnd)




    def KD_optimize(self):
        view_ys, _ = self.feature_extraction(self.view_xs)
        view_ys.append(self.fusion_feature)

        # Online Knowledge Distillation
    
    def gLS_optimize(self):

        self.tensor_matrix()


        plt.figure()
        self.fig, self.ax = plt.subplots()
        testaccs = []
        # label guided isa feature extraction
        for epoch in range(self.epochsL):
            self.epochL = epoch
            print('------------------------------------ epochsL: ' + str(epoch) + ' ---------------------------------------')
            #self.Y_optimize()
            self.Y_optimize(soft_flag = True)
            self.VCL_optimize()
            
            testaccs = testaccs+self.IR_optimize()

            self.R_optimize()
            #self.Y_optimize()
            #self.ICL_optimize() # multi-model feature contrastive learning

            print(self.testdata_evaluate()[0])

            #self.VCL_optimize() # single-model W contrastive learning

        print(testaccs)
        print(np.arange(len(testaccs)))
        print(testaccs)
        self.ax.set_xticks(np.arange(len(testaccs)))
        self.ax.plot(np.arange(len(testaccs)), testaccs)
        plt.savefig(self.path+"loss.png", dpi=500)
        #plt.show()

        # contrastive learning on Representation Layer
        #print('&&&&&&&&&&&&&&&&&&&&& contrastive learning &&&&&&&&&&&&&&&&&&&&&&&&&')
        #self.VCL_optimize()
        self.testdata_evaluate()

        # knowledge distillation among view representations and fusion representation
        #self.KD_optimize()
        
        # cat

        # classification


def main():
    #dataset: Orl.mat  (min_max:0-245)  COIL20.mat  (min_max:0-1)  PIE_pose27.mat (min_max:0-255)   USPS7000.mat
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1, metavar='N',
                        help='input id of gpu [default: -1 (use CPU)]')
    parser.add_argument('--dataset', type=int, default=2, metavar='N')
    parser.add_argument('--epochsR', type=int, default=100, metavar='N', 
                        help='epochs of label guided isa feature extraction optimization')
    parser.add_argument('--epochsY', type=int, default=3, metavar='N',
                        help='epochs of isa loss optimization')
    parser.add_argument('--epochsC', type=int, default=30, metavar='N',
                        help='epochs of contrastive loss optimization')
    parser.add_argument('--epochsL', type=int, default=10, metavar='N', 
                        help='loop epochs of R and Y')
    parser.add_argument('--max-iter', type=int, default=10, metavar='N')
    parser.add_argument('--nodes', type=int, default=800, metavar='N')
    parser.add_argument('--p', type=float, default=2.0, metavar='N')
    parser.add_argument('--lrR', type=float, default=1e-4, metavar='N', help='if p==2: lrR better be 1e-3, else if p==0.7: lrR better be 1e-4 or smaller')#
    parser.add_argument('--lrY', type=float, default=1e-6, metavar='N') # cannot be too small
    #parser.add_argument('--lrY', type=float, default=1e-3, metavar='N')
    parser.add_argument('--lrz', type=float, default=1e-3, metavar='N')

    args = parser.parse_args()

    assert not (args.p < 2 and args.lrR > 1e-4)

    print(args)
    cuda_idx = args.gpu#0 for gpu & 1 for cpu
    cuda_flag = False
    if cuda_idx == -1:
        cuda_flag = False
    else:
        cuda_flag = True
        torch.cuda.set_device(cuda_idx)
    seed = 100

    epochsR = args.epochsR
    epochsY = args.epochsY
    from utils import read_matlab_data_clustering
    filenames = [
        'yale_mtv.mat',
        'scene-15.mat', 
        'Caltech101-all-4view.mat', 
        'handwritten.mat', 
        'PIE_face_10.mat', #true
        'cub_googlenet_doc2vec_c10.mat', 
        'ORL_mtv.mat', 
        'flower17.mat', 
        'yaleB_mtv.mat', 
        'bbc_sports-dense.mat', #10
        'NUSWIDEOBJ-half.mat', 
        'BBC-dense.mat', 
        'Reuters.mat', 
        'RBPpred.mat', 
        'MSRC.mat' 
    ]
    filename = filenames[args.dataset-1]
    print(filename)
    T = False
    Ts = [1, 5, 6, 9, 10, 12]
    if args.dataset in Ts:
        T = True
    #X, gnd, view_number = read_matlab_data_clustering('/Users/ezio/Documents/Python/database-mtv/', filename, bShuffle = False, T = T)
    #X, gnd, view_number = read_matlab_data_clustering('/Users/ezio/Documents/Python/database-mtv/', filename, bShuffle = True, T = T)


    foldK = 5
    for whichTest in range(1, foldK+1):
        whichDataset = args.dataset

        # get KFold traindata and testdata
        traindata, testdata, view_number = utils.read_matlab_data_kfold('/home/ezio/dataset-mtv/', filename, K=foldK, whichTest=whichTest, bShuffle=True, Normal=2,which=whichDataset)
        #traindata, testdata, view_number = utils.read_matlab_data_kfold('../database-mtv/', filename, K=foldK, whichTest=whichTest, bShuffle=True, Normal=2,which=whichDataset)
        
        n_class = len(np.unique(traindata.labels))  #  Orl: 40,  COIL20: 20, PIE_pose27: 68,  USPS7000: 10
        num_of_one_class = int(len(traindata.labels)/n_class)
        print('n_class')
        print(n_class)
        print('num_of_one_class')
        print(num_of_one_class)



        out = args.nodes
        num_of_sum = 2 
        mid = out * num_of_sum
        print('mid:')
        print(mid)
        print('out:')
        print(out)
        
        view_blocks = []
        blocks = []

        import os
        path = filename[:-4]+'/'+str(foldK)+'-'+str(whichTest)+'/'

        if not os.path.exists(path):
            os.makedirs(path)
         
        Y = np.array([])

        NEWTRAIN = True
        if NEWTRAIN:
            for v in range(view_number):
                b = ISA_block(W=None, Xshape = traindata.data_list[v].shape, n_mid=mid, n_out=out, num_of_sum=num_of_sum,  p=args.p, cuda_flag = cuda_flag)

                view_blocks.append(b)

            #!
            b = ISA_block(W=None, Xshape = (traindata.data_list[0].shape[0], out*view_number), n_mid=n_class * 2, n_out=n_class, num_of_sum=num_of_sum, p=args.p, cuda_flag = cuda_flag)
            #b = ISA_block(W=None, Xshape = (traindata.data_list[0].shape[0], 80*view_number), n_mid=n_class * 2, n_out=n_class, num_of_sum=num_of_sum, p=args.p, cuda_flag = cuda_flag)
            blocks.append(b)

        else:
            m = sio.loadmat(path+'ISA-'+params+'.mat')
            Y = m['Y']
            gnd = m['gnd'][0]


        ISA_model = ISA_clustering(
            view_blocks, 
            blocks, 
            out, 
            max_iter = args.max_iter, 
            lrY = args.lrY, 
            lrR = args.lrR,
            epochsR = args.epochsR, 
            epochsY = args.epochsY, 
            epochsC = args.epochsC, 
            epochsL = args.epochsL,
            cuda_flag = cuda_flag, 
            #batch_size=num_of_one_class
            batch_size=len(traindata.labels),
            traindata = traindata,
            testdata = testdata,
            path = path
        )

        ISA_model.gLS_optimize()

        #Y = ISA_model.Y
        testacc, pred = ISA_model.testdata_evaluate()
        print('final result of fold '+str(whichTest))
        print(str(testacc))

        #sio.savemat(path+'/Y.mat',{'Y':Y, 'gnd':gnd})

        #print('classifying result of Y')
        #acc, loss, pred = ISA_model.evaluate(Y, gnd)

        file = open(path+'pred-'+str(whichTest)+'.txt', 'w+')
        file.write('pred_labels:\n[')
        #file.write(str(pred))
        for e in pred:
            file.write(str(e) + ', ')
        file.write(']\n')
        file.write('real_labels:\n[')
        for e in testdata.labels:
            file.write(str(e) + ', ')
        file.write(']\n')
        file.close()

        #print('clustering result of Y')
        #cluster(Y, gnd, path=path+'/y-clustering.txt')

        #print('final result of Z')
        #cluster(torch.abs(Z) + torch.abs(Z.T), gnd, affinity='precomputed', path=path+'/zlabels-'+path+'.txt')


if __name__=='__main__':
    main()
