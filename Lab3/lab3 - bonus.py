import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

def TestModel(c):
    data=np.load('friends.npy',allow_pickle=True)
    MF1 = c.MakeMFMatrix(c.F1, c.nlen)
    MF2 = c.MakeMFMatrix(c.F2, c.nlen1)
    labels = ["Arabic", "Chinese","Czech", "Dutch", "English", "French", "German", "Greek", "Irish", "Italian", "Japanese", "Korean", "Polish", "Portuguese", "Russian", "Scottish", "Spanish", "Vietnamese"]
    score=0
    for i in range(data.item()['X'].shape[1]):
        P = c.ForwardPass(data.item()['X'][:, [i]], MF1, MF2)
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
        self.n1=20
        self.k1=5 
        self.n2=20
        self.k2=3
        self.batch=100
        self.epoch=10
        self.d=dataset['d']
        self.nlen=dataset['nlen']
        self.K=dataset['classes']
        self.nlen1=self.nlen - self.k1 + 1
        self.nlen2=self.nlen1 - self.k2 + 1
        self.mu=0
        self.sigma=np.sqrt(2 / dataset['d'])
        np.random.seed(112) # for the best model in lab
        # np.random.seed(1) # for first two models in lab
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
        self.CheckGradient=False
        self.bias1=np.random.normal(0, self.sigma,((((self.nlen - self.k1 + 1) * self.n1),1)))
        self.bias2=np.random.normal(0, self.sigma,((((self.nlen1 - self.k2 + 1) * self.n2),1)))
        self.bias3=np.random.normal(0, self.sigma,(((self.dataset['classes']),1)))
        self.bias1_momentum=np.zeros(self.bias1.shape)
        self.bias2_momentum=np.zeros(self.bias2.shape)
        self.bias3_momentum=np.zeros(self.bias3.shape)

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

    def computeGradientsNum(self, X, Y, MF1, MF2):
        h = 1e-5
        grad_F1 = np.zeros(self.F1.shape)
        grad_W = np.zeros(self.W.shape)
        grad_F2 = np.zeros(self.F2.shape)
        grad_b1= np.zeros(self.bias1.shape)
        grad_b2= np.zeros(self.bias2.shape)
        grad_b3= np.zeros(self.bias3.shape)
        print("Numerical F1 gradient")
        for i in (range(self.F1.shape[2])):
            for j in range(self.F1.shape[1]):
                for k in range(self.F1.shape[0]):
                    F1_try = np.copy(self.F1)
                    F1_try[k, j, i] -= h
                    MF1_try = self.MakeMFMatrix(F1_try, self.nlen)
                    l1 = self.ComputeCost(X, Y, MF1_try, MF2)
                    F1_try = np.copy(self.F1)
                    F1_try[k, j, i] += h
                    MF1_try = self.MakeMFMatrix(F1_try, self.nlen)
                    l2 = self.ComputeCost(X, Y, MF1_try, MF2)
                    grad_F1[k, j, i] = (l2 - l1) / (2 * h)

        print("Numerical F2 gradient")
        for i in (range(self.F2.shape[2])):
            for j in range(self.F2.shape[1]):
                for k in range(self.F2.shape[0]):
                    F2_try = np.copy(self.F2)
                    F2_try[k, j, i] -= h
                    MF2_try = self.MakeMFMatrix(F2_try, self.nlen1)
                    l1 = self.ComputeCost(X, Y, MF1, MF2_try)
                    F2_try = np.copy(self.F2)
                    F2_try[k, j, i] += h
                    MF2_try = self.MakeMFMatrix(F2_try, self.nlen1)
                    l2 = self.ComputeCost(X, Y, MF1, MF2_try)
                    grad_F2[k, j, i] = (l2 - l1) / (2 * h)

        print("Numerical W gradient")
        W_bk=np.copy(self.W)
        for i in (range(self.W.shape[0])):
            for j in range(self.W.shape[1]):
                W_try = np.copy(W_bk)
                W_try[i][j] -= h
                self.W=np.copy(W_try)
                l1 = self.ComputeCost(X, Y, MF1, MF2)
                W_try = np.copy(W_bk)
                W_try[i][j] += h
                self.W=np.copy(W_try)
                l2 = self.ComputeCost(X, Y, MF1, MF2)
                grad_W[i, j] = (l2 - l1) / (2 * h)
        self.W=np.copy(W_bk)
        
        print("Numerical Bias1 gradient")
        bias_bk=np.copy(self.bias1)
        for i in range(len(self.bias1)):
            bias_try=np.copy(self.bias1)
            bias_try[i]+=h
            self.bias1=np.copy(bias_try)
            l1 = self.ComputeCost(X, Y, MF1, MF2)
            bias_try=np.copy(bias_bk)
            bias_try[i]-=h
            self.bias1=np.copy(bias_try)
            l2 = self.ComputeCost(X, Y, MF1, MF2)
            grad_b1[i] = (l2-l1) / (2 * h)
        self.bias1=np.copy(bias_bk)

        print("Numerical Bias2 gradient")
        bias_bk=np.copy(self.bias2)
        for i in range(len(self.bias2)):
            bias_try=np.copy(self.bias2)
            bias_try[i]+=h
            self.bias2=np.copy(bias_try)
            l1 = self.ComputeCost(X, Y, MF1, MF2)
            bias_try=np.copy(bias_bk)
            bias_try[i]-=h
            self.bias2=np.copy(bias_try)
            l2 = self.ComputeCost(X, Y, MF1, MF2)
            grad_b2[i] = (l2-l1) / (2 * h)
        self.bias2=np.copy(bias_bk)

        print("Numerical Bias3 gradient")
        bias_bk=np.copy(self.bias3)
        for i in range(len(self.bias3)):
            bias_try=np.copy(self.bias3)
            bias_try[i]+=h
            self.bias3=np.copy(bias_try)
            l1 = self.ComputeCost(X, Y, MF1, MF2)
            bias_try=np.copy(bias_bk)
            bias_try[i]-=h
            self.bias3=np.copy(bias_try)
            l2 = self.ComputeCost(X, Y, MF1, MF2)
            grad_b3[i] = (l2-l1) / (2 * h)
        self.bias3=np.copy(bias_bk)


        return grad_F1, grad_F2, grad_W ,grad_b1,grad_b2,grad_b3

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
        D=MF1.dot(X)+self.bias1
        X1=self.relu(D)
        D=MF2.dot(X1)+self.bias2
        X2=self.relu(D)
        S=self.W.dot(X2)+self.bias3
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
        gradbias1=np.zeros((self.bias1.shape))
        gradbias2=np.zeros((self.bias2.shape))
        gradbias3=np.zeros((self.bias3.shape))


        D=MF1.dot(X)+self.bias1
        X1=self.relu(D)
        D=MF2.dot(X1)+self.bias2
        X2=self.relu(D)
        S=self.W.dot(X2)+self.bias3
        P = self.softmax(S)
        
        G = -(Y.T - P.T).T
        gradW = np.dot(G, X2.T) / X2.shape[1]
        gradbias3 = np.reshape(1/X2.shape[1] * G.dot(np.ones(X2.shape[1])), (Y.shape[0], 1))
        
        n = X1.shape[1]
        G = np.dot(G.T, self.W)
        S2 = np.where(X2 > 0, 1, 0)
        G = np.multiply(G.T, S2)
        gradbias2=np.reshape(1/n * G.dot(np.ones(n)), (X2.shape[0], 1))
        
        
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
        gradbias1=np.reshape(1/n * G.dot(np.ones(n)), (X1.shape[0], 1))
        for j in (range(n)):
            gj = G[:, [j]]
            xj = X[:, [j]]
            Mj = self.MakeMXMatrix(xj, self.dataset['d'], self.k1)
            a = gj.shape[0]
            gj = gj.reshape((int(a / self.n1), self.n1))
            v = np.dot(Mj.T, gj)
            gradF1 += v.reshape(self.F1.shape, order='F') / n
        return gradW,gradF1,gradF2,gradbias1,gradbias2,gradbias3

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
        plt.title('Confusion Matrix')
        plt.show()
        
    def modelPerfomance(self):
        class_idx, counts = np.unique(self.dataset['Labels'][self.dataset['TrainIndex']], return_counts=True)
        X=self.dataset['X_train']
        Y=self.dataset['Y_train']
        MF1=self.MakeMFMatrix(self.F1, self.nlen)
        MF2 = self.MakeMFMatrix(self.F2, self.nlen1)
        self.createConfusionMatrix(X, Y, MF1, MF2)
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
        MF1=self.MakeMFMatrix(self.F1, self.nlen)
        MF2 = self.MakeMFMatrix(self.F2, self.nlen1)
  
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

                gradW,gradF1,gradF2,gradbias1,gradbias2,gradbias3=self.ComputeGradients(Xbatch, Ybatch,MF1,MF2)

                if(self.CheckGradient==True and j==0 and i==0):
                
                    gradW_a,gradF1_a,gradF2_a,gradbias1_a,gradbias2_a,gradbias3_a = self.ComputeGradients(Xbatch, Ybatch,MF1,MF2)
                    self.gradbias1_a=gradbias1_a
                    self.gradbias2_a=gradbias2_a
                    self.gradbias3_a=gradbias3_a

                    grad_F1_n, grad_F2_n, grad_W_n,gradbias1_n,gradbias2_n,gradbias3_n=self.computeGradientsNum(Xbatch, Ybatch,MF1,MF2)
                    self.gradbias1_n=gradbias1_n
                    self.gradbias2_n=gradbias2_n
                    self.gradbias3_n=gradbias3_n
                    
                    print('Mean Difference between Gradients of Weights :',np.mean(gradW_a-grad_W_n))
                    print('Mean Difference between Gradients of F1 :',np.mean(gradF1_a-grad_F1_n))
                    print('Mean Difference between Gradients of F2 :',np.mean(gradF2_a-grad_F2_n))
                    print('Mean Difference between Gradients of Bias1 :',np.mean(gradbias1_a-gradbias1_n))
                    print('Mean Difference between Gradients of Bias2 :',np.mean(gradbias2_a-gradbias2_n))
                    print('Mean Difference between Gradients of Bias3 :',np.mean(gradbias3_a-gradbias3_n))

                    
                self.W_momentum = self.W_momentum * self.rho + self.eta * gradW
                self.F2_momentum = self.F2_momentum * self.rho + self.eta * gradF2
                self.F1_momentum = self.F1_momentum * self.rho + self.eta * gradF1
                self.bias1_momentum=self.bias1_momentum* self.rho + self.eta * gradbias1
                self.bias2_momentum=self.bias2_momentum* self.rho + self.eta * gradbias2
                self.bias3_momentum=self.bias3_momentum* self.rho + self.eta * gradbias3
                self.F1 -= self.F1_momentum
                self.F2 -= self.F2_momentum
                self.W -= self.W_momentum
                self.bias1-=self.bias1_momentum
                self.bias2-=self.bias2_momentum
                self.bias3-=self.bias3_momentum
                MF1=self.MakeMFMatrix(self.F1, self.nlen)
                MF2 = self.MakeMFMatrix(self.F2, self.nlen1)
            valLoss = self.ComputeCost(X_Val, Y_val,MF1,MF2)
            vallacc = self.computeAccuracy(X_Val,self.dataset['Labels'][self.dataset['ValIndex']])
            trainloss=self.ComputeCost(X,Y,MF1,MF2)
            self.TrainingLoss.append(trainloss)
            self.AccuracyVal.append(vallacc)
            self.ValidationLoss.append(valLoss)
dataset=DataPreparation()
c=CNN(dataset)
c.fit()
# c.modelPerfomance()
# TestModel(c)
