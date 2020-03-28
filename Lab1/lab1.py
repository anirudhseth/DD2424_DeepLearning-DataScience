import pickle 
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def load_batch(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        x, y = data['data'], np.array(data['labels'])
        x = x.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    return x, y

def loadDataset(data_batch):
    xtrain = []
    ytrain = []
    d= 'cifar-10-batches-py'
    for i in range(data_batch):
        x=i+1
        f = os.path.join(d, 'data_batch_%d' % x)
        x, y = load_batch(f)
        xtrain.append(x)
        ytrain.append(y)
    xtrain = np.concatenate(xtrain)
    ytrain = np.concatenate(ytrain)
    xtest, ytest = load_batch(os.path.join(d, 'test_batch'))
    xtrain = np.reshape(xtrain, (xtrain.shape[0], -1))

    xtest = np.reshape(xtest, (xtest.shape[0], -1))

    x_mean_train = np.mean(xtrain, axis=0)
    xtrain -= x_mean_train
    xtrain=xtrain/255.0
    xtest -= x_mean_train
    xtest=xtest/255.0    
    return xtrain.T, ytrain, xtest.T, ytest

class neuralNet():

    def plotWeights(self):
        w = self.W.reshape(10, 32, 32, 3)
        w_min, w_max = np.min(w), np.max(w)
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            # Rescale the weights to be between 0 and 255 for image representation
            w_img = 255.0 * (w[i].squeeze() - w_min) / (w_max - w_min)
            plt.imshow(w_img.astype('uint8'))
            plt.axis('off')
            plt.title(classes[i])
        plt.show()

    def __init__(self,k,d):
        self.W = np.random.normal(loc=0.0, scale=0.01, size=(k, d))
        self.b= np.random.normal(loc=0.0, scale=0.01,size= (k, 1))
        self.k=k
        self.d=d
        self.loss_training=[]
        self.loss_test=[]
        self.accuracy_training=[]
        self.accuracy_test=[]
                
    def softmax(self,x):
        softmax = np.exp(x) / sum(np.exp(x))
        return softmax

    def EvaluateClassifier(self,X):
        s = np.dot(self.W, X) + self.b
        P = self.softmax(s)
        return P

    def ComputeGradsNum(self,loss,X, Y, P,lamda, h):
        """ Converted from matlab code """
        grad_W = np.zeros(self.W.shape)
        grad_b = np.zeros(self.b.shape)
        c = self.computeCost(X, Y.T,lamda,loss)[0]
        for i in range(len(self.b)):
            tempB=neuralNet(self.k,self.d)
            tempB.b = np.array(self.b)
            tempB.W=self.W
            tempB.b[i] += h
            c2 = tempB.computeCost(X, Y.T,lamda,loss)
            grad_b[i] = (c2-c) / h
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                tempW=neuralNet(self.k,self.d)
                tempW.W = np.array(self.W)
                tempW.b=self.b
                tempW.W[i,j] += h
                c2 = tempW.computeCost(X, Y.T,lamda,loss)
                grad_W[i,j] = (c2-c) / h
        return [grad_W, grad_b]

    def ComputeAccuracy(self,X,y):
        y_Acc=np.dot(self.W,X)
        y_Acc_pred=np.argmax(y_Acc, axis=0)
        accuracy = np.mean(y == y_Acc_pred)
        return accuracy

    def oneHotVector(self,y,size=10):
        oneHot=[]
        for i in y:
            t=np.zeros(size)
            t[i]=1.0
            oneHot.append(t)
        return np.asarray(oneHot, dtype=np.float32)

    def computeCost(self,X, Y,lambda_reg,loss):
        regularization = lambda_reg * np.sum(np.square(self.W))
        loss_sum = 0
        for i in range(X.shape[1]):
            x = np.zeros((3072, 1))
            y = np.zeros((10, 1))
            x = X[:, [i]]
            y = Y[:, [i]]
            loss_sum += self.calculateLoss(x, y,loss)
        loss_sum /= X.shape[1]
        final = loss_sum + regularization
        return final

    def calculateLoss(self,x, y,loss):
        if(loss=='crossEntropy'):
            l = - np.log(np.dot(y.T, self.EvaluateClassifier(x)))[0]
        elif(loss=='SVM'):
            # s = np.dot(self.W,x)+self.b
            # y_int = np.where(y.T[0] == 1)[0][0]
            # margins=np.maximum(0,s-s[y_int]+1)
            # margins[y_int]=0
            # l=np.sum(margins)
            s = np.dot(self.W, x) + self.b
            l = 0
            y_int = np.where(y.T[0] == 1)[0][0]
            for j in range(10):
                if j != y_int:
                    l += max(0, s[j] - s[y_int] + 1)
        return l

    def compute_gradients(self,X, Y, P,lambda_reg,loss):
        if(loss=='crossEntropy'):
            G = -(Y - P.T).T
            return (np.dot(G,X)) / X.shape[0] + 2 * lambda_reg * self.W, np.mean(G, axis=-1, keepdims=True)
        elif(loss=='SVM'):
            n = X.shape[0]
            gradW = np.zeros((self.k, self.d))
            gradb = np.zeros((self.k, 1))
            for i in range(n):
                x = X[i, :]
                y_int = np.where(Y[i, :].T == 1)[0][0]
                s = np.dot(self.W, x.reshape(-1,1)) + self.b
                for j in range(self.k):
                    if j != y_int:
                        if max(0, s[j] - s[y_int] + 1) != 0:
                            gradW[j] += x
                            gradW[y_int] += -x
                            gradb[j, 0] += 1
                            gradb[y_int, 0] += -1

            gradW /= n
            gradW += lambda_reg * self.W
            gradb /= n
            return gradW, gradb


    def fit(self,loss,x_train,y_train,x_test,y_test,epoch,eta,lamda,batch,shuffle,decay,gradient):
        for i in range(epoch):
            if(shuffle):
                x_train,y_train=self.randomShuffle(x_train,y_train)
            n = x_train.shape[1]
            n_batch = int(np.floor(n / batch))
            for j in range(n_batch):
                j_start = int(j * n_batch)
                j_end = int((j + 1) *n_batch)
                if j == n_batch - 1:
                    j_end = n
                Xbatch = x_train[:, j_start:j_end]
                Ybatch = y_train[j_start:j_end]
                Pbatch = self.EvaluateClassifier(Xbatch)
                grad_W, grad_b = self.compute_gradients(Xbatch.T, self.oneHotVector(Ybatch), Pbatch,lamda,loss)
                if(gradient==True and j==0 and i==0):
                    h=1e-6
                    Pgrad = self.EvaluateClassifier(Xbatch[:,0:50])
                    grad_W_anal, grad_b_anal = self.compute_gradients(Xbatch[:,0:50].T, self.oneHotVector(Ybatch[0:50]), Pgrad,lamda,loss)
                    grad_W_num,grad_b_num=self.ComputeGradsNum(loss,Xbatch[:,0:50], self.oneHotVector(Ybatch[0:50]), Pgrad,lamda, h)
                    print('Mean Differnce between Gradients of Weights:',np.mean(grad_W_anal-grad_W_num))
                    print('Mean Differnce between Gradients of Bias:',np.mean(grad_b_anal-grad_b_num))
                self.W -= eta * grad_W
                self.b -= eta * grad_b
            J=self.computeCost(x_train, self.oneHotVector(y_train).T,lamda,loss)
            J_test=self.computeCost(x_test, self.oneHotVector(y_test).T,lamda,loss)
            self.loss_training.append(J)
            self.loss_test.append(J_test)
            Accuracy_train=self.ComputeAccuracy(x_train,y_train)
            self.accuracy_training.append(Accuracy_train)
            Accuracy_test=self.ComputeAccuracy(x_test,y_test)
            self.accuracy_test.append(Accuracy_test)
            if(decay):
                eta=eta*0.9
                # print('ETA Decay:',eta)
                
        print('Epoch:',epoch,' ETA:',eta,' Lambda: ',lamda,' Batch Size:',batch)
        print('Training Size:',x_train.shape[1])
        print('Test Size:',x_test.shape[1])
        self.performance()

    def performance(self):
        print('Accuracy on Test Set:'+str(self.accuracy_test[-1]))
        print('Accuracy on Training Set:'+str(self.accuracy_training[-1]))
        plt.plot(self.loss_training,label="Training Set")
        plt.plot(self.loss_test,label="Test Set")
        plt.legend()
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.show()
        plt.plot(self.accuracy_training,label="Training Set")
        plt.plot(self.accuracy_test,label="Test Set")
        plt.legend()
        plt.ylabel('Accuracy')
        plt.xlabel('epochs')
        plt.show()
        self.plotWeights()
        
    def randomShuffle(self,x,y):
        index=np.arange(np.shape(y)[0])
        np.random.shuffle(index)
        return x[:,index],y[index]
            




x_train, y_train, x_test, y_test = loadDataset(data_batch=1)
d=np.shape(x_train)[0]
k=10
loss='SVM'

# [epoch,eta,lamda,batch]=[10,0.1,0,100]

# a=neuralNet(k,d)
# a.fit(loss,x_train, y_train, x_test, y_test,epoch,eta,lamda,batch,shuffle=False,decay=False,gradient=False)

# [epoch,eta,lamda,batch]=[40,0.001,0,100]
# b=neuralNet(k,d)
# b.fit(loss,x_train, y_train, x_test, y_test,epoch,eta,lamda,batch,shuffle=False,decay=False,gradient=False)

[epoch,eta,lamda,batch]=[10,0.01,.1,100]
c=neuralNet(k,d)
c.fit(loss,x_train, y_train, x_test, y_test,epoch,eta,lamda,batch,shuffle=False,decay=False,gradient=False)

# [epoch,eta,lamda,batch]=[40,0.001,1,100]
# d2=neuralNet(k,d)
# d2.fit(loss,x_train, y_train, x_test, y_test,epoch,eta,lamda,batch,shuffle=False,decay=False,gradient=False)





