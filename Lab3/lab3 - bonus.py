import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

def TestModel(c):
    data=np.load('friends.npy',allow_pickle=True)
    MF=[]
    for f in range(c.numLayer):
        MF_temp=c.MakeMFMatrix(c.F[f], c.nlen[f])
        MF.append(MF_temp)
    labels = ["Arabic", "Chinese","Czech", "Dutch", "English", "French", "German", "Greek", "Irish", "Italian", "Japanese", "Korean", "Polish", "Portuguese", "Russian", "Scottish", "Spanish", "Vietnamese"]
    score=0
    for i in range(data.item()['X'].shape[1]):
        P = c.ForwardPass(data.item()['X'][:, [i]], MF)
        label = np.argmax(P)
        p=np.max(P)
        print('Last Name: '+data.item()['Names'][i] + "\tActual Label: "+labels[data.item()['Labels'][i]] + '\tPredicted Label:'+labels[label]+'\tProb:'+str(p))
        if(labels[data.item()['Labels'][i]]==labels[label]):
            score+=1
    print('Accuracy is ',score/data.item()['X'].shape[1])

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
        self.numLayer=2
        self.n=[50,50] #specify number of filter in each layer here
        self.k=[7,3]
        self.batch=100
        self.epoch=200
        self.d=dataset['d']
        self.nlen=[]
        self.nlen.append(dataset['nlen'])
        self.K=dataset['classes']
        for i in range(self.numLayer):
            self.nlen.append(self.nlen[-1]-self.k[i]+1)
        self.mu=0
        self.sigma=np.sqrt(2 / dataset['d'])
        np.random.seed(112) 
        self.F=[]
        for i in range(self.numLayer):
            if i==0:
                F_i=np.random.normal(self.mu, self.sigma, (self.d,self.k[i],self.n[i]))
            else:
                F_i=np.random.normal(self.mu, self.sigma, (self.n[i-1],self.k[i],self.n[i]))
            self.F.append(F_i)

        self.W = np.random.normal(self.mu, self.sigma, (self.K,self.n[-1]*self.nlen[-1]))
        self.ValidationLoss=[]
        self.TrainingLoss=[]
        self.AccuracyVal=[]
        self.F_mom=[]
        for i in range(self.numLayer):
                F_i=np.zeros(self.F[i].shape)
                self.F_mom.append(F_i)

        self.W_momentum = np.zeros(self.W.shape)
        self.Imbalance=True
        self.CheckGradient=False
        self.bias=[]
        for i in range(self.numLayer):
            bias_temp_i=np.random.normal(0, self.sigma,((((self.nlen[i] - self.k[i] + 1) * self.n[i]),1)))
            self.bias.append(bias_temp_i)

        self.biasW=np.random.normal(0, self.sigma,(((self.dataset['classes']),1)))
        self.bias_mom=[]
        for i in range(self.numLayer):
            bias_temp_i=np.zeros(self.bias[i].shape)
            self.bias_mom.append(bias_temp_i)
        self.biasW_momentum=np.zeros(self.biasW.shape)
        self.dropoutFlag=True
        self.dropoutProb=0
        self.CyclicEta=True
        self.eta_min=0.001
        self.eta_max=0.1
        self.ns=500
        self.etaPlot=[]
    
    def getETA(self,t):
        if t <= self.ns:
            eta = self.eta_min + t/self.ns * (self.eta_max - self.eta_min)
        elif t <= 2*self.ns:
            eta = self.eta_max - (t - self.ns)/self.ns * (self.eta_max - self.eta_min)
        t = (t+1) % (2*self.ns)
        return eta,t

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

    def dropout(self,X, drop_probability):
        #https://stats.stackexchange.com/questions/205932/dropout-scaling-the-activation-versus-inverting-the-dropout
        keep_probability = 1 - drop_probability
        mask = np.random.rand(*X.shape) < keep_probability
        if keep_probability > 0.0:
            scale = (1/keep_probability)
        else:
            scale = 0.0
        return mask * X * scale

    def ForwardPass(self,X,MF):
        X_hidden=[]
        X_hidden.append(X)
        for f in range(self.numLayer):
            D=MF[f].dot(X_hidden[-1])+self.bias[f]
            X_temp=self.relu(D)
            X_hidden.append(X_temp)
        S=self.W.dot(X_hidden[-1])+self.biasW
        if (self.dropoutFlag==True):
            S=self.dropout(S, self.dropoutProb)
        P=self.softmax(S)
        return P

    def ComputeCost(self, X, Y,MF):
        N = X.shape[1]
        P = self.ForwardPass(X,MF)
        loss = -1/N *np.sum(Y*np.log(P))
        return loss
    
    def ComputeGradients(self, X, Y,MF):
        
        gradF=[]
        for f in range(self.numLayer):
            gradF.append(np.zeros((self.F[f].shape)))

        gradW = np.zeros((self.W.shape))
        gradbias=[]
        for f in range(self.numLayer):
            gradbias.append(np.zeros((self.bias[f].shape)))

        gradbiasW=np.zeros((self.biasW.shape))
        
        
        X_hidden=[]
        X_hidden.append(X)
        for f in range(self.numLayer):
            D=MF[f].dot(X_hidden[-1])+self.bias[f]
            X_temp=self.relu(D)
            X_hidden.append(X_temp)
        S=self.W.dot(X_hidden[-1])+self.biasW
        if (self.dropoutFlag==True):
            S=self.dropout(S, self.dropoutProb)
        P=self.softmax(S)

        G = -(Y.T - P.T).T
        gradW = np.dot(G, X_hidden[-1].T) / X_hidden[-1].shape[1]
        gradbiasW = np.reshape(1/X_hidden[-1].shape[1] * G.dot(np.ones(X_hidden[-1].shape[1])), (Y.shape[0], 1))
        

        for f in range(self.numLayer,0,-1):
            # print(f)
            n = X_hidden[f-1].shape[1]
            if(f==self.numLayer):
                G = np.dot(G.T, self.W)
            else:
                G = np.dot(G.T, MF[f])
            S=np.where(X_hidden[f] > 0, 1, 0)
            G = np.multiply(G.T, S)
            gradbias[f-1]=np.reshape(1/n * G.dot(np.ones(n)), (X_hidden[f].shape[0], 1))
            for j in (range(n)):
                xj = X_hidden[f-1][:, [j]]
                gj = G[:, [j]]
                if(f==1):
                    MjGen = self.MakeMXMatrix(xj, self.dataset['d'], self.k[f-1])
                else:
                    MjGen = self.MakeMXMatrix(xj, self.n[f-2], self.k[f-1])
                a = gj.shape[0]
                gj = gj.reshape((int(a / self.n[f-1]), self.n[f-1]))
                v2 = np.dot(MjGen.T, gj)
                gradF[f-1] += v2.reshape(self.F[f-1].shape, order='F') / n
            


        # n = X_hidden[2].shape[1]
        # G = np.dot(G.T, self.W)
        # S3 = np.where(X_hidden[3] > 0, 1, 0)
        # G = np.multiply(G.T, S3)
        # gradbias[2]=np.reshape(1/n * G.dot(np.ones(n)), (X_hidden[3].shape[0], 1))
        # for j in (range(n)):
        #     xj = X_hidden[2][:, [j]]
        #     gj = G[:, [j]]
        #     MjGen = self.MakeMXMatrix(xj, self.n[1], self.k[2])
        #     a = gj.shape[0]
        #     gj = gj.reshape((int(a / self.n[2]), self.n[2]))
        #     v2 = np.dot(MjGen.T, gj)
        #     gradF[2] += v2.reshape(self.F[2].shape, order='F') / n

        # G = np.dot(G.T, MF[2])
        # S2 = np.where(X_hidden[2] > 0, 1, 0)
        # G = np.multiply(G.T, S2)
        # n = X_hidden[1].shape[1]
        # gradbias[1]=np.reshape(1/n * G.dot(np.ones(n)), (X_hidden[2].shape[0], 1))
        # for j in (range(n)):
        #     gj = G[:, [j]]
        #     xj = X_hidden[1][:, [j]]
        #     Mj = self.MakeMXMatrix(xj, self.n[0], self.k[1])
        #     a = gj.shape[0]
        #     gj = gj.reshape((int(a / self.n[1]), self.n[1]))
        #     v = np.dot(Mj.T, gj)
        #     gradF[1] += v.reshape(self.F[1].shape, order='F') / n

        # G = np.dot(G.T, MF[1])
        # S1 = np.where(X_hidden[1] > 0, 1, 0)
        # G = np.multiply(G.T, S1)
        # n = X_hidden[0].shape[1]
        # gradbias[0]=np.reshape(1/n * G.dot(np.ones(n)), (X_hidden[1].shape[0], 1))
        # for j in (range(n)):
        #     gj = G[:, [j]]
        #     xj = X_hidden[0][:, [j]]
        #     Mj = self.MakeMXMatrix(xj, self.dataset['d'], self.k[0])
        #     a = gj.shape[0]
        #     gj = gj.reshape((int(a / self.n[0]), self.n[0]))
        #     v = np.dot(Mj.T, gj)
        #     gradF[0] += v.reshape(self.F[0].shape, order='F') / n


        return gradW,gradF,gradbias,gradbiasW

    def computeAccuracy(self, X, Y):
        MF=[]
        for f in range(self.numLayer):
            MF_temp=self.MakeMFMatrix(self.F[f], self.nlen[f])
            MF.append(MF_temp)
        acc = 0
        for i in range(X.shape[1]):
            P = self.ForwardPass(X[:, [i]], MF)
            label = np.argmax(P)
            if label == Y[i]:
                acc += 1
        acc /= X.shape[1]
        return acc

    def createConfusionMatrix(self, X, Y, MF):
        P = self.ForwardPass(X, MF)
        P = np.argmax(P, axis=0)
        T = np.argmax(Y, axis=0)
        from sklearn.metrics import confusion_matrix
        plt.pcolormesh(confusion_matrix(P, T))
        plt.title('Confusion Matrix')
        plt.show()
        
    def modelPerfomance(self):
        class_idx, counts = np.unique(self.dataset['Labels'][self.dataset['TrainIndex']], return_counts=True)
        X=self.dataset['X_train']
        Y=self.dataset['Y_train']
        MF=[]
        for f in range(self.numLayer):
            MF_temp=self.MakeMFMatrix(self.F[f], self.nlen[f])
            MF.append(MF_temp)
        self.createConfusionMatrix(X, Y,MF)
        print('Training Accuracy:',self.computeAccuracy(X,self.dataset['Labels'][self.dataset['TrainIndex']]))
        print('Validation Accuracy:',self.AccuracyVal[-1])
        plt.plot(self.AccuracyVal,label="Validation Set")
        plt.xlabel('Steps')
        plt.ylabel('Accuracy')
        plt.show()
        plt.plot(self.ValidationLoss,label="Validation Set")
        # plt.plot(self.TrainingLoss,label="Training Set")
        plt.legend()
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.show()

    def fit(self):

        if(self.CyclicEta):
            t=0
            self.eta=self.eta_min
        else:
            self.eta=self.eta_min
        X=self.dataset['X_train']
        Y=self.dataset['Y_train']
        X_Val=self.dataset['X_Val']
        Y_val=self.dataset['Y_Val']
        if(self.Imbalance==True):
            class_idx, counts = np.unique(self.dataset['Labels'][self.dataset['TrainIndex']], return_counts=True)
            minClass=np.min(counts)
            n_batch = int(np.floor((minClass * self.K) / self.batch))
        else:
            n = X.shape[1]
            n_batch = int(np.floor(n / self.batch))
        
        MF=[]
        for f in range(self.numLayer):
            MF_temp=self.MakeMFMatrix(self.F[f], self.nlen[f])
            MF.append(MF_temp)

        for i in tqdm(range(self.epoch)):

            if(self.Imbalance==True):
                idx=[]
                TrainingLabels=self.dataset['Labels'][self.dataset['TrainIndex']]
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

                gradW,gradF,gradbias,gradbiasW=self.ComputeGradients(Xbatch, Ybatch,MF)


                self.W_momentum = self.W_momentum * self.rho + self.eta * gradW
                self.biasW_momentum=self.biasW_momentum* self.rho + self.eta * gradbiasW
                for f in range(self.numLayer):
                    self.F_mom[f] = self.F_mom[f] * self.rho + self.eta * gradF[f]
                    self.bias_mom[f]=self.bias_mom[f]* self.rho + self.eta * gradbias[f]


                self.W -= self.W_momentum
                self.biasW-=self.biasW_momentum
                for f in range(self.numLayer):
                    self.F[f] -= self.F_mom[f]
                    self.bias[f]-=self.bias_mom[f]
                
                MF=[]
                for f in range(self.numLayer):
                    MF_temp=self.MakeMFMatrix(self.F[f], self.nlen[f])
                    MF.append(MF_temp)
                if(self.CyclicEta):
                    self.eta,t=self.getETA(t)
                    self.etaPlot.append(self.eta)
            valLoss = self.ComputeCost(X_Val, Y_val,MF)
            vallacc = self.computeAccuracy(X_Val,self.dataset['Labels'][self.dataset['ValIndex']])
            trainloss=self.ComputeCost(X,Y,MF)
            self.TrainingLoss.append(trainloss)
            self.AccuracyVal.append(vallacc)
            self.ValidationLoss.append(valLoss)
dataset=DataPreparation()
c=CNN(dataset)
c.fit()
print(c.AccuracyVal[-1])
c.modelPerfomance()
TestModel(c)
