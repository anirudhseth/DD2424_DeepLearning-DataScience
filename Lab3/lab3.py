import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse
import matplotlib.pyplot as plt
def ExtractNames():
    data = pd.read_csv('ascii_names.txt', header = None)
    arr=np.asarray(data[0][:])
    y=[]
    names=[]
    maxLen=0
    for i in range(arr.shape[0]):
        line=arr[i]
        line=line.split()
        nm=''.join(line[0:-1]).lower()
        names.append(nm)
        if(maxLen<len(nm)):
            maxLen=len(nm)
        y.append(int(line[-1])-1)
    return np.asarray(names),np.asarray(y),maxLen

def DataPreparation():
    X,Y,nlen=ExtractNames()
    c=''.join(set(' '.join(X[:])))
    c=''.join(sorted(c))
    d=len(c)
    classes=len(np.unique(Y))

    oneHot_X = np.zeros((d * nlen, X.shape[0]))

    char_to_int = dict((t, i) for i, t in enumerate(c))
    index = 0
    for name in X:
        one_hot = np.zeros((d, nlen))
        integer_encoded = [char_to_int[char] for char in name]
        i = 0
        for value in integer_encoded:
            letter = np.zeros((d))
            letter[value] = 1
            one_hot[:, i] = letter
            i += 1
        oneHot_X[:d*nlen, index] = one_hot.flatten('F')
        index += 1

    oneHot_Y = np.zeros((Y.shape[0], classes))
    for i in range(Y.shape[0]):
        oneHot_Y[i][Y[i]] = 1
    oneHot_Y=oneHot_Y.T



    Valindex=np.loadtxt('Validation_Inds.txt',dtype=int)
    Valindex-=1
    TrainIndex=list(set(np.arange(X.shape[0]))-set(Valindex))


    oneHot_X_Val=oneHot_X[:,Valindex]
    oneHot_X_Train=oneHot_X[:,TrainIndex]
    oneHot_Y_Train=oneHot_Y[:,TrainIndex]
    oneHot_Y_Val=oneHot_Y[:,Valindex]

    dataset={}
    dataset['Names']=X
    dataset['Labels']=Y
    dataset['X_train']=oneHot_X_Train
    dataset['Y_train']=oneHot_Y_Train
    dataset['X_Val']=oneHot_X_Val
    dataset['Y_Val']=oneHot_Y_Val
    dataset['ValIndex']=Valindex
    dataset['TrainIndex']=TrainIndex
    dataset['C']=c
    dataset['d']=d
    dataset['nlen']=nlen
    dataset['classes']=classes

    return dataset

class CNN():
    def __init__(self,dataset):
        self.dataset=dataset
        self.eta=0.01
        self.rho=0.9
        self.n1=50
        self.k1=5
        self.n2=50
        self.k2=3
        self.batch=100
        self.epoch=2000
        self.d=dataset['d']
        self.nlen=dataset['nlen']
        self.K=dataset['classes']
        self.nlen1=self.nlen - self.k1 + 1
        self.nlen2=self.nlen1 - self.k2 + 1
        self.mu=0
        self.sigma=np.sqrt(2 / dataset['d'])
        np.random.seed(1)
        self.F1 = np.random.normal(self.mu, self.sigma, (self.d,self.k1,self.n1))
        self.F2 = np.random.normal(self.mu, self.sigma, (self.n1,self.k2,self.n2))
        self.W = np.random.normal(self.mu, self.sigma, (self.K,self.n2*self.nlen2))
        self.ValidationLoss=[]
        self.TrainingLoss=[]
        self.AccuracyVal=[]
        self.F1_momentum = np.zeros(self.F1.shape)
        self.F2_momentum = np.zeros(self.F2.shape)
        self.W_momentum = np.zeros(self.W.shape)
        self.Imbalance=True

    def softmax(self,x):
        softmax = np.exp(x) / sum(np.exp(x))
        return softmax   

    def MakeMFMatrix(self,F,nlen):
        dd,k,nf=F.shape
        MF = np.zeros(((nlen - k + 1) * nf, nlen * dd))
        VF = F.reshape((dd * k, nf), order='F').T
        for i in range(nlen - k + 1):
            MF[i * nf:(i + 1) * nf, dd * i:dd * i + dd * k] = VF
        return MF


    def MakeMXMatrix(self, x_input, d, k):
        n_len = int(x_input.shape[0] / d)
        VX = np.zeros((n_len - k + 1, k * d))
        x_input = x_input.reshape((d, n_len), order='F')
        for i in range(n_len - k + 1):
            VX[i, :] = (x_input[:, i:i + k].reshape((k * d, 1), order='F')).T
        return VX

    def relu(self,t):
        return np.maximum(0,t)

    def ForwardPass(self,X,MF1,MF2):
        H1=MF1.dot(X)
        H1=self.relu(H1)
        H2=MF2.dot(H1)
        H2=self.relu(H2)
        S=self.W.dot(H2)
        P=self.softmax(S)
        return P
    
    def ComputeCost(self, X, Y,MF1,MF2):
        N = X.shape[1]
        P = self.ForwardPass(X,MF1,MF2)
        loss = -1/N *np.sum(Y*np.log(P))
        return loss
    
    def ComputeGradients(self, X, Y,MF1,MF2):

        gradF1 = np.zeros((self.F1.shape))
        gradF2 = np.zeros((self.F2.shape))
        gradW = np.zeros((self.W.shape))

        dot = np.dot(MF1, X)
        X1 = np.where(dot > 0, dot, 0)
        dot = np.dot(MF2, X1)
        X2 = np.where(dot > 0, dot, 0)
        S = np.dot(self.W, X2)
        P = self.softmax(S)
        
        G = -(Y.T - P.T).T
        gradW = np.dot(G, X2.T) / X2.shape[1]

        G = np.dot(G.T, self.W)
        S2 = np.where(X2 > 0, 1, 0)
        G = np.multiply(G.T, S2)

        n = X1.shape[1]
        for j in (range(n)):
            xj = X1[:, [j]]
            gj = G[:, [j]]
            MjGen = self.MakeMXMatrix(xj, self.n1, self.k2)
            a = gj.shape[0]
            gj = gj.reshape((int(a / self.n2), self.n2))
            v2 = np.dot(MjGen.T, gj)
            gradF2 += v2.reshape(self.F2.shape, order='F') / n

        G = np.dot(G.T, MF2)
        S1 = np.where(X1 > 0, 1, 0)
        G = np.multiply(G.T, S1)
        n = X.shape[1]
        for j in (range(n)):
            gj = G[:, [j]]
            xj = X[:, [j]]
            Mj = self.MakeMXMatrix(xj, dataset['d'], self.k1)
            a = gj.shape[0]
            gj = gj.reshape((int(a / self.n2), self.n2))
            v = np.dot(Mj.T, gj)
            gradF1 += v.reshape(self.F1.shape, order='F') / n
        return gradW,gradF1,gradF2

    def computeAccuracy(self, X, Y):
        MF1 = self.MakeMFMatrix(self.F1, self.nlen)
        MF2 = self.MakeMFMatrix(self.F2, self.nlen1)
        acc = 0
        for i in range(X.shape[1]):
            P = self.ForwardPass(X[:, [i]], MF1, MF2)
            label = np.argmax(P)
            if label == Y[i]:
                acc += 1
        acc /= X.shape[1]
        return acc

    def createConfusionMatrix(self, X, Y, MF1, MF2):
        P = self.ForwardPass(X, MF1, MF2)
        P = np.argmax(P, axis=0)
        T = np.argmax(Y, axis=0)
        from sklearn.metrics import confusion_matrix
        plt.pcolormesh(confusion_matrix(P, T))
        plt.show()

    def fit(self,dataset):
        X=dataset['X_train']
        Y=dataset['Y_train']
        X_Val=dataset['X_Val']
        Y_val=dataset['Y_Val']
        if(self.Imbalance==True):
            class_idx, counts = np.unique(dataset['Labels'][dataset['TrainIndex']], return_counts=True)
            minClass=np.min(counts)
            n_batch = int(np.floor((minClass * self.K) / self.batch))
        else:
            n = X.shape[1]
            n_batch = int(np.floor(n / self.batch))
        MF1=self.MakeMFMatrix(self.F1, self.nlen)
        MF2 = self.MakeMFMatrix(self.F2, self.nlen1)
  
        for i in tqdm(range(self.epoch)):
            if(self.Imbalance==True):
                idx=[]
                TrainingLabels=dataset['Labels'][dataset['TrainIndex']]
                for c in range(self.K):
                    classIndex=np.where(TrainingLabels==c)[0]
                    idx+=list(np.random.choice(classIndex, minClass))
                np.random.shuffle(idx)

            for j in (range(n_batch)): 
                j_start = int(j * self.batch)
                j_end = int((j + 1) *self.batch)
                
                if j == n_batch - 1:
                    if(self.Imbalance==True):
                        j_end = len(idx)
                    else:

                        j_end = n
                if(self.Imbalance==True):
                    idxFix = idx[j_start: j_end]
                    Xbatch = X[:, idxFix]
                    Ybatch = Y[:,idxFix]
                else:
                    idx = np.arange(j_start, j_end)
                    Xbatch = X[:, idx]
                    Ybatch = Y[:,idx]



                gradW,gradF1,gradF2=self.ComputeGradients(Xbatch, Ybatch,MF1,MF2)
                self.W_momentum = self.W_momentum * self.rho + self.eta * gradW
                self.F2_momentum = self.F2_momentum * self.rho + self.eta * gradF2
                self.F1_momentum = self.F1_momentum * self.rho + self.eta * gradF1
                self.F1 -= self.F1_momentum
                self.F2 -= self.F2_momentum
                self.W -= self.W_momentum
                MF1=self.MakeMFMatrix(self.F1, self.nlen)
                MF2 = self.MakeMFMatrix(self.F2, self.nlen1)
            valLoss = self.ComputeCost(X_Val, Y_val,MF1,MF2)
            vallacc = self.computeAccuracy(X_Val,dataset['Labels'][dataset['ValIndex']])
            self.AccuracyVal.append(vallacc)
            self.ValidationLoss.append(valLoss)
        # print('Training',self.computeAccuracy(X,dataset['Labels'][dataset['TrainIndex']]))
        self.createConfusionMatrix(X, Y, MF1, MF2)
dataset=DataPreparation()
c=CNN(dataset)
c.fit(dataset)

