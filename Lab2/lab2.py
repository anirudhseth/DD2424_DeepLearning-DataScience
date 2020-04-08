import pickle 
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def loadMultipleBatches(numOfBatches,ValidationSize):
    for i in np.arange(numOfBatches):
        filename="cifar-10-batches-py/data_batch_"+str(i+1)
        if(i==0):
            x_train,y_train=load_batch(filename)
        else:
            tempX,tempY=load_batch(filename)
            x_train=np.concatenate((x_train,tempX),axis=1)
            y_train=np.concatenate((y_train,tempY))
    x_validation = x_train[:, -ValidationSize:]
    y_validation = y_train[-ValidationSize:]
    x_train = x_train[:, :-ValidationSize]
    y_train = y_train[:-ValidationSize]
    x_test, y_test= load_batch("cifar-10-batches-py/test_batch")
    return x_train,y_train,x_test, y_test,x_validation,y_validation

def load_batch(filename):
    with open(filename, 'rb') as f:
        dataDict = pickle.load(f, encoding='bytes')
        x = (dataDict[b"data"]).T
        x_mean = np.mean(x, axis=1, keepdims=True)
        x_std = np.std(x, axis=1, keepdims=True)
        x = (x - x_mean) / x_std
        y = np.array(dataDict[b"labels"])
    return x, y

def unpickle(filename):
    with open(filename, 'rb') as f:
        file_dict = pickle.load(f, encoding='bytes')
    return file_dict

class neuralNet():
    def __init__(self,k,d,m):
    
        np.random.seed(123)
        self.W1 = np.random.normal(loc=0.0, scale=1/np.sqrt(d), size=(m, d))
        self.W2 = np.random.normal(loc=0.0, scale=1/np.sqrt(m), size=(k, m))
        self.b1= np.zeros((m,1))
        self.b2= np.zeros((k,1))
        self.etaPlot=[]
        self.k=k
        self.d=d
        self.m=m
        self.loss_training=[]
        self.loss_test=[]
        self.loss_validation=[]
        self.cost_training=[]
        self.cost_test=[]
        self.cost_validation=[]
        self.accuracy_training=[]
        self.accuracy_test=[]
        self.accuracy_validation=[]
        self.dropoutFlag=False 

    def softmax(self,x):
        softmax = np.exp(x) / sum(np.exp(x))
        return softmax

    def oneHotVector(self,y,size=10):
        oneHot=[]
        for i in y:
            t=np.zeros(size)
            t[i]=1.0
            oneHot.append(t)
        return np.asarray(oneHot, dtype=np.float32).T

    def relu(self,t):
        return np.maximum(0,t)

    def EvaluateClassifier(self,X):
        s1 = self.W1.dot(X) + self.b1
        h=self.relu(s1)
        if(self.dropoutFlag==True):
            h=self.dropout(h, 0.5)
        s=self.W2.dot(h)+self.b2
        if(self.dropoutFlag==True):
            s=self.dropout(s,0.5)
        p = self.softmax(s)
        return h,p


    def dropout(self,X, drop_probability):
        #https://stats.stackexchange.com/questions/205932/dropout-scaling-the-activation-versus-inverting-the-dropout
        keep_probability = 1 - drop_probability
        mask = np.random.rand(*X.shape) < keep_probability
        if keep_probability > 0.0:
            scale = (1/keep_probability)
        else:
            scale = 0.0
        return mask * X * scale

    def ComputeGradsNum(self,X, Y,lamda, h):
        """ Converted from matlab code """
        grad_W1 = np.zeros(self.W1.shape)
        grad_b1 = np.zeros(self.b1.shape)
        grad_W2 = np.zeros(self.W2.shape)
        grad_b2 = np.zeros(self.b2.shape)
        c,_ = self.computeCost(X, Y,lamda)
        for i in range(len(self.b1)):
            tempB1=neuralNet(self.k,self.d,self.m)
            tempB1.b1 = np.array(self.b1)
            tempB1.W1=self.W1
            tempB1.b1[i] += h
            c2,_ = tempB1.computeCost(X, Y,lamda)
            grad_b1[i] = (c2-c) / h
        for i in range(self.W1.shape[0]):
            for j in range(self.W1.shape[1]):
                tempW1=neuralNet(self.k,self.d,self.m)
                tempW1.W1 = np.array(self.W1)
                tempW1.b1=self.b1 
                tempW1.W1[i,j] += h
                c2,_ = tempW1.computeCost(X, Y,lamda)
                grad_W1[i,j] = (c2-c) / h
        
        for i in range(len(self.b2)):
            tempB2=neuralNet(self.k,self.d,self.m)
            tempB2.b2 = np.array(self.b2)
            tempB2.W2=self.W2
            tempB2.b2[i] += h
            c2,_ = tempB2.computeCost(X, Y,lamda)
            grad_b2[i] = (c2-c) / h
        for i in range(self.W2.shape[0]):
            for j in range(self.W2.shape[1]):
                tempW2=neuralNet(self.k,self.d,self.m)
                tempW2.W2 = np.array(self.W2)
                tempW2.b2=self.b2
                tempW2.W2[i,j] += h
                c2,_ = tempW2.computeCost(X, Y,lamda)
                grad_W2[i,j] = (c2-c) / h
        return [grad_W1, grad_b1,grad_W2, grad_b2]

    def ComputeAccuracy(self,X,y):
        Y_Scores=self.EvaluateClassifier(X)[1]
        Y_prediction=np.argmax(Y_Scores, axis=0)
        accuracy = np.mean(y == Y_prediction)
        return accuracy

    def computeCost(self,X, Y,lambda_reg):
        N = X.shape[1]
        Y=self.oneHotVector(Y)
        P = self.EvaluateClassifier(X)[1]
        loss = -1/N *np.sum(Y*np.log(P))
        cost = loss + lambda_reg * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
        return  cost,loss

    def compute_gradients(self,X, Y,lambda_reg):
        N=X.shape[1]
        Y=self.oneHotVector(Y)
        H,P = self.EvaluateClassifier(X)
        G = - (Y - P)
        grad_W2 = 1/N * G.dot(H.T)  + 2 * lambda_reg * self.W2
        grad_b2 = np.reshape(1/N * G.dot(np.ones(N)), (Y.shape[0], 1))
        G = (self.W2.T).dot(G)
        G= np.multiply(G, H > 0)
        grad_W1 = 1/N * G.dot(X.T) + lambda_reg * self.W1
        grad_b1 = np.reshape(1/N *G.dot(np.ones(N)), (self.m, 1))
        return grad_W1,grad_b1,grad_W2,grad_b2

    def augmentBatchData(self,X,Y):
        augmentPercentage=0.1
        augmentsize=int(augmentPercentage*X.shape[1])
        randindex=np.random.randint(0,X.shape[1],augmentsize)
        X_aug=X[:,randindex]
        whiteNoise=np.random.normal(loc=0.0, scale=0.0001, size=X_aug.shape)
        X_aug+=whiteNoise
        Y_aug=Y[randindex]
        X=np.hstack((X,X_aug))
        Y=np.hstack((Y,Y_aug))
        return X,Y

    def getETA(self,t):
        if t <= self.n_s:
            eta = self.eta_min + t/self.n_s * (self.eta_max - self.eta_min)
        elif t <= 2*self.n_s:
            eta = self.eta_max - (t - self.n_s)/self.n_s * (self.eta_max - self.eta_min)
        t = (t+1) % (2*self.n_s)
        return eta,t

    def randomShuffle(self,x,y):
        index=np.arange(np.shape(y)[0])
        np.random.shuffle(index)
        return x[:,index],y[index]

    def fit(self,x_train,y_train,x_test,y_test,x_validation, y_validation,epoch,eta_min,eta_max,n_s,lamda,batch,shuffle=False,gradient=False,log=False,CyclicEta=True,augment=False,dropoutFlag=False,momentum=False):
        if(shuffle):
            x_train,y_train=self.randomShuffle(x_train,y_train)
        if(dropoutFlag==True):
            self.dropoutFlag=True
        if(momentum==True):
            self.b1_mom=np.zeros(self.b1.shape) 
            self.b2_mom=np.zeros(self.b2.shape)
            self.W1_mom=np.zeros(self.W1.shape)
            self.W2_mom=np.zeros(self.W2.shape)      
            self.p=0.95     
        if(CyclicEta):
            t=0
            self.eta_min=eta_min
            self.eta_max=eta_max
            self.n_s=n_s
            eta=self.eta_min
            eta=eta_min
        else:
            eta=eta_min
        for i in range(epoch):
            if(shuffle):
                x_train,y_train=self.randomShuffle(x_train,y_train)
            n = x_train.shape[1]
            n_batch = int(np.floor(n / batch))
            for j in range(batch):
                j_start = int(j * n_batch)
                j_end = int((j + 1) *n_batch)
                if j == n_batch - 1:
                    j_end = n
                Xbatch = x_train[:, j_start:j_end]
                Ybatch = y_train[j_start:j_end]
                if(augment==True):
                    Xbatch,Ybatch=self.augmentBatchData(Xbatch,Ybatch)
                grad_W1,grad_b1,grad_W2,grad_b2 = self.compute_gradients(Xbatch, Ybatch,lamda)
                if(gradient==True and j==0 and i==0):
                    h=1e-5
                    grad_W1,grad_b1,grad_W2,grad_b2 = self.compute_gradients(Xbatch[:,0:20], Ybatch[0:20],lamda)
                    grad_W1_num,grad_b1_num,grad_W2_num,grad_b2_num=self.ComputeGradsNum(Xbatch[:,0:20], Ybatch[0:20],lamda, h)
                    print('Mean Differnce between Gradients of Weights (1):',np.mean(grad_W1-grad_W1_num))
                    print('Mean Differnce between Gradients of Bias (1):',np.mean(grad_b1-grad_b1_num))
                    print('Mean Differnce between Gradients of Weights (2):',np.mean(grad_W2-grad_W2_num))
                    print('Mean Differnce between Gradients of Bias (2):',np.mean(grad_b2-grad_b2_num))
                if(momentum==True):
                    self.W1_mom = self.p*self.W1_mom+eta*grad_W1
                    self.b1_mom = self.p*self.b1_mom+eta*grad_b1
                    self.W2_mom = self.p*self.W2_mom+eta*grad_W2
                    self.b2_mom = self.p*self.b2_mom+eta*grad_b2

                    self.W1 -= eta * grad_W1
                    self.b1 -= eta * grad_b1
                    self.W2 -= eta * grad_W2
                    self.b2 -= eta * grad_b2
                else:
                    self.W1 -= eta * grad_W1
                    self.b1 -= eta * grad_b1
                    self.W2 -= eta * grad_W2
                    self.b2 -= eta * grad_b2
                if(CyclicEta):
                    eta,t=self.getETA(t)
                    self.etaPlot.append(eta)
    


            J,L=self.computeCost(x_train, y_train,lamda)
            J_test,L_test=self.computeCost(x_test, y_test,lamda)
            J_val,L_val=self.computeCost(x_validation, y_validation,lamda)
            self.cost_training.append(J)
            self.cost_test.append(J_test)
            self.cost_validation.append(J_val)
            self.loss_training.append(L)
            self.loss_test.append(L_test)
            self.loss_validation.append(L_val)
            Accuracy_train=self.ComputeAccuracy(x_train,y_train)
            self.accuracy_training.append(Accuracy_train)
            Accuracy_test=self.ComputeAccuracy(x_test,y_test)
            self.accuracy_test.append(Accuracy_test)
            Accuracy_validation=self.ComputeAccuracy(x_validation,y_validation)
            self.accuracy_validation.append(Accuracy_validation)

        # print to log file
        if(log==True):
            import calendar,time
            st = str(calendar.timegm(time.gmtime()))
            logfile='Plots/log_'+st+'.log'
            log = open(logfile, "a")
            bak=sys.stdout
            sys.stdout = log    
            print('Epoch:',epoch,' ETA(Cyclic):',self.eta_min,'-',eta_max,' Lambda: ',lamda,' Batch Size:',batch)
            print('Training Size:',x_train.shape[1])
            print('Test Size:',x_test.shape[1])
            print('Validation Size:',x_validation.shape[1])
            self.performance(st)
            log.close()
            sys.stdout = bak

    def performance(self,st):
        plt.rcParams["figure.figsize"] = (7,5)
        print('Accuracy on Test Set:'+str(self.accuracy_test[-1]))
        print('Accuracy on Training Set:'+str(self.accuracy_training[-1]))
        print('Accuracy on Validation Set:'+str(self.accuracy_validation[-1]))
        plt.plot(self.loss_training,label="Training Set")
        # plt.plot(self.loss_test,label="Test Set")
        plt.plot(self.loss_validation,label="Validation Set")
        plt.legend()
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.savefig('Plots/loss_'+st)
        plt.show()
        
        plt.plot(self.cost_training,label="Training Set")
        # plt.plot(self.cost_test,label="Test Set")
        plt.plot(self.cost_validation,label="Validation Set")
        plt.legend()
        plt.ylabel('cost')
        plt.xlabel('epochs')
        plt.savefig('Plots/cost_'+st)
        plt.show()
        plt.plot(self.accuracy_training,label="Training Set")
        # plt.plot(self.accuracy_test,label="Test Set")
        plt.plot(self.accuracy_validation,label="Validation Set")
        plt.legend()
        plt.ylabel('Accuracy')
        plt.xlabel('epochs')
        plt.savefig('Plots/accuracy_'+st)
        plt.show()
        # plt.plot(self.etaPlot)
        # plt.ylabel('ETA')
        # plt.xlabel('steps')
        # plt.savefig('Plots/ETA_'+st)
        # plt.show()
      
    def randomShuffle(self,x,y):
        index=np.arange(np.shape(y)[0])
        np.random.shuffle(index)
        return x[:,index],y[index]
            



# x_train, y_train =load_batch("cifar-10-batches-py/data_batch_1")
# x_test, y_test= load_batch("cifar-10-batches-py/test_batch")
# x_validation,y_validation=load_batch("cifar-10-batches-py/data_batch_2")
# d=np.shape(x_train)[0]
# k=10
# m=50
############### Figure 3 ###############

# [epoch,eta_min,eta_max,ns,lamda,batch]=[10,1e-5,1e-1,500,0.01,100]
# nn=neuralNet(k,d,m)
# nn.fit(x_train, y_train, x_test, y_test,x_validation, y_validation,epoch,eta_min,eta_max,ns,lamda,batch)

############### Figure 4 ###############

# [epoch,eta_min,eta_max,ns,lamda,batch]=[48,1e-5,1e-1,800,0.01,100]
# nn2=neuralNet(k,d,m)
# nn2.fit(x_train, y_train, x_test, y_test,x_validation, y_validation,epoch,eta_min,eta_max,ns,lamda,batch)

############### Search Ld ###############
# x_train, y_train,x_test,y_test,x_validation,y_validation=loadMultipleBatches(5,5000)
# d=np.shape(x_train)[0]
# k=10
# m=50



# import random
# l_min=1e-01
# l_max=1e-05
# l=np.sort(np.random.uniform(l_min,l_max,16))
# accuracy_coarse=[]
# # for ld in l
# accuracy=[]
# [epoch,eta_min,eta_max,ns,batch]=[80,1e-5,1e-1,1000,100]
# for lamda in l:
#     nn_search=neuralNet(k,d,m)
#     nn_search.fit(x_train, y_train, x_test, y_test,x_validation, y_validation,epoch,eta_min,eta_max,ns,lamda,batch)
#     temp=nn_search.accuracy_validation
#     if(len(accuracy)==0):
#         accuracy=temp
#     else:
#         accuracy=np.vstack((accuracy,temp))

# dict={}
# dict['lambda']=l
# dict['accuracy']=accuracy
# np.save("Plots/BroadSearch.npy", dict)

# a=np.amax(accuracy,axis=1)
# plt.scatter(l,a)
# plt.title('Coarse Random Search')
# plt.ylabel('Accuracy(Validation Set)')
# plt.xlabel('lambda')
# plt.show()


# l_min=0.00599464
# l_max=0.00925069
# l=np.linspace(l_min,l_max,8)
# accuracy_coarse=[]
# # for ld in l
# accuracy2=[]
# [epoch,eta_min,eta_max,ns,batch]=[80,1e-5,1e-1,1000,100]
# for lamda in l:
#     nn_search=neuralNet(k,d,m)
#     nn_search.fit(x_train, y_train, x_test, y_test,x_validation, y_validation,epoch,eta_min,eta_max,ns,lamda,batch)
#     temp=nn_search.accuracy_validation
#     if(len(accuracy2)==0):
#         accuracy2=temp
#     else:
#         accuracy2=np.vstack((accuracy2,temp))

# dict={}
# dict['lambda']=l
# dict['accuracy']=accuracy2
# np.save("Plots/NarrowSearch.npy", dict)

# a=np.amax(accuracy2,axis=1)
# plt.scatter(l,a)
# plt.title('Fine Random Search')
# plt.ylabel('Accuracy(Validation Set)')
# plt.xlabel('lambda')
# plt.show()

# [epoch,eta_min,eta_max,ns,lamda,batch]=[100,1e-5,1e-1,1000,0.00878554,100]
# nnBest=neuralNet(k,d,m)
# nnBest.fit(x_train, y_train, x_test, y_test,x_validation, y_validation,epoch,eta_min,eta_max,ns,lamda,batch,log=True)


############### Search ETA ###############
# x_train, y_train,x_test,y_test,x_validation,y_validation=loadMultipleBatches(5,5000)
# d=np.shape(x_train)[0]
############### Model 1 ###############
# k=10
# m=80
# etaTrend=[]
# eta_=[1e-9]
# for i in range(20):
#     if(eta_[-1]<1):
#         # print(eta_[-1])
#         [epoch,eta_min,eta_max,ns,lamda,batch]=[8,1e-5,1e-5,800,0.00878554,100]
#         nnETA=neuralNet(k,d,m)
#         nnETA.fit(x_train, y_train, x_test, y_test,x_validation, y_validation,epoch,eta_[-1],eta_[-1],ns,lamda,batch,CyclicEta=False)
#         etaTrend.append(nnETA.accuracy_validation[-1])
#         # print('acc',nnETA.accuracy_validation[-1])
#         eta_.append(eta_[-1]*4)
# plt.rcParams["figure.figsize"] = (7,5)
# plt.scatter(eta_[:15],etaTrend[:15])
# plt.plot(eta_[:15],etaTrend[:15])
# plt.xlabel('ETA')
# plt.ylabel('Accuracy(Validation Set)')
# plt.show()

# [epoch,eta_min,eta_max,ns,lamda,batch]=[48,0.000262144,0.067108864,800,0.00878554,100]
# nnETAModel1=neuralNet(k,d,m)
# nnETAModel1.fit(x_train, y_train, x_test, y_test,x_validation, y_validation,epoch,eta_min,eta_max,ns,lamda,batch,log=True)

############### Model 2 ###############
# k=10
# m=40
# etaTrend=[]
# eta_=[1e-9]
# for i in range(20):
#     if(eta_[-1]<1):
#         # print(eta_[-1])
#         [epoch,eta_min,eta_max,ns,lamda,batch]=[40,1e-5,1e-5,1000,0.00878554,100]
#         nnETA=neuralNet(k,d,m)
#         nnETA.fit(x_train, y_train, x_test, y_test,x_validation, y_validation,epoch,eta_[-1],eta_[-1],ns,lamda,batch,CyclicEta=False)
#         etaTrend.append(nnETA.accuracy_validation[-1])
#         print('acc',nnETA.accuracy_validation[-1])
#         eta_.append(eta_[-1]*3)
# plt.rcParams["figure.figsize"] = (7,5)
# plt.scatter(eta_[:19],etaTrend[:19])
# plt.plot(eta_[:19],etaTrend[:19])
# plt.xlabel('ETA')
# plt.ylabel('Accuracy(Validation Set)')
# plt.show()

# [epoch,eta_min,eta_max,ns,lamda,batch]=[60,0.0047829690000000015,0.12914016300000003,1000,0.00599464,100]
# nnETAModel2=neuralNet(k,d,m)
# nnETAModel2.fit(x_train, y_train, x_test, y_test,x_validation, y_validation,epoch,eta_min,eta_max,ns,lamda,batch,log=False)

############### Augmentation ###############
# k=10
# m=80
# x_train, y_train,x_test,y_test,x_validation,y_validation=loadMultipleBatches(5,5000)
# d=np.shape(x_train)[0]

# [epoch,eta_min,eta_max,ns,lamda,batch]=[48,0.000262144,0.067108864,800,0.00599464,100]
# nnAugment=neuralNet(k,d,m)
# nnAugment.fit(x_train, y_train, x_test, y_test,x_validation, y_validation,epoch,eta_min,eta_max,ns,lamda,batch,log=True,augment=True)

############### Dropout ###############

# k=10
# m=200
# x_train, y_train,x_test,y_test,x_validation,y_validation=loadMultipleBatches(5,5000)
# d=np.shape(x_train)[0]

# [epoch,eta_min,eta_max,ns,lamda,batch]=[60,0.0047829690000000015,0.12914016300000003,1000,0.00599464,100]
# nnDropout=neuralNet(k,d,m)
# nnDropout.fit(x_train, y_train, x_test, y_test,x_validation, y_validation,epoch,eta_min,eta_max,ns,lamda,batch,log=False,augment=False,dropoutFlag=True)

############### Momentum and Everything Combined ###############
#average models
# k=10
# m=80
# x_train, y_train,x_test,y_test,x_validation,y_validation=loadMultipleBatches(5,5000)
# d=np.shape(x_train)[0]

# [epoch,eta_min,eta_max,ns,lamda,batch]=[100,0.000262144,0.067108864,1000,0.00599464,500]
# nnMom=neuralNet(k,d,m)
# nnMom.fit(x_train, y_train, x_test, y_test,x_validation, y_validation,epoch,eta_min,eta_max,ns,lamda,batch,shuffle=True,log=True,augment=True,dropoutFlag=False,momentum=True)

# k=10
# m=100
# x_train, y_train,x_test,y_test,x_validation,y_validation=loadMultipleBatches(5,5000)
# d=np.shape(x_train)[0]

# [epoch,eta_min,eta_max,ns,lamda,batch]=[100,0.000262144,0.067108864,1000,0.00599464,500]
# nnMom2=neuralNet(k,d,m)
# nnMom2.fit(x_train, y_train, x_test, y_test,x_validation, y_validation,epoch,eta_min,eta_max,ns,lamda,batch,shuffle=True,log=True,augment=True,dropoutFlag=False,momentum=True)

#good models
# k=10
# m=120
# x_train, y_train,x_test,y_test,x_validation,y_validation=loadMultipleBatches(5,5000)
# d=np.shape(x_train)[0]

# [epoch,eta_min,eta_max,ns,lamda,batch]=[100,0.000262144,0.067108864,1000,0.00599464,500]
# nnMom3=neuralNet(k,d,m)
# nnMom3.fit(x_train, y_train, x_test, y_test,x_validation, y_validation,epoch,eta_min,eta_max,ns,lamda,batch,shuffle=True,log=True,augment=True,dropoutFlag=False,momentum=True)


# k=10
# m=140
# x_train, y_train,x_test,y_test,x_validation,y_validation=loadMultipleBatches(5,5000)
# d=np.shape(x_train)[0]

# [epoch,eta_min,eta_max,ns,lamda,batch]=[100,0.000262144,0.067108864,1000,0.00599464,500]
# nnMom4=neuralNet(k,d,m)
# nnMom4.fit(x_train, y_train, x_test, y_test,x_validation, y_validation,epoch,eta_min,eta_max,ns,lamda,batch,shuffle=True,log=True,augment=True,dropoutFlag=False,momentum=True)
