import sys
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import OrderedDict
from tqdm import tqdm

def softmax(x):
    sf = np.exp(x) / sum(np.exp(x))
    return sf




class RNN:
    def __init__(self):
        np.random.seed(123)
        self.K=93
        self.m=100
        self.eta=0.1
        self.seq_length=10
        self.epsilon=np.finfo(float).eps
        sig=.01
        mean=0
        self.params={
            "U":np.random.normal(mean, sig, size=(self.m, self.K)),
            "W":np.random.normal(mean, sig, size=(self.m, self.m)),
            "V":np.random.normal(mean, sig, size=(self.K, self.m)),
            "b":np.zeros((self.m,1)),
            "c":np.zeros((self.K,1))

        }
        self.mTheta= {  "W": np.zeros_like(self.params['W']),
                        "U": np.zeros_like(self.params['U']),
                        "V": np.zeros_like(self.params['V']),
                        "b": np.zeros_like(self.params['b']),
                        "c": np.zeros_like(self.params['c'])
                  }

        self.prevH=np.zeros((self.m,1))
        self.trueLoss=[]
        self.smoothLoss=[]
        self.memory=[]
        
        

    def forward(self,inputs):
        n=len(inputs)
        prevH=np.copy(self.prevH)
        for t in range(n):
            x=np.zeros((self.K,1))
            x[inputs[t]]=1
            self.at=self.params['W'].dot(prevH)+self.params['U'].dot(x)+self.params['b']
            self.ht=np.tanh(self.at)
            self.ot=self.params['V'].dot(self.ht)+self.params['c']
            self.pt=softmax(self.ot)
            prevH=np.copy(self.ht)
            dict= { 
            "at": self.at,
            "ht": self.ht,
            "ot": self.ot,
            "pt": self.pt}
            self.memory.append(dict)
        
    
    def clipGradients(self):
        for grad in self.grads:
            self.grads[grad] = np.clip(self.grads[grad], -5, 5)

    def adaGrad(self):
        for theta in self.params:
            self.mTheta[theta]+=np.square(self.grads[theta])
            self.params[theta]-=(self.eta*self.grads[theta])/(np.sqrt(self.mTheta[theta] +
                    np.finfo(float).eps))
    
    def backward(self,inputs,targets):
        steps=len(inputs)
        self.grads={
            "U":np.zeros((self.m, self.K)),
            "W":np.zeros((self.m, self.m)),
            "V":np.zeros((self.K, self.m)),
            "b":np.zeros((self.m,1)),
            "c":np.zeros((self.K,1)),
            "a":np.zeros_like(self.memory[0]['at']),
            "h":np.zeros_like(self.memory[0]['ht']),
            "o":np.zeros_like(self.memory[0]['ot']),
            "hpp":np.zeros_like(self.memory[0]['ht'])
        }

        for t in reversed(range(steps)):
            xt=np.zeros((self.K,1))
            yt=np.zeros((self.K,1))
            yt[targets[t]]=1
            xt[[inputs[t]]]=1
            pt=self.memory[t]['pt']
            ht=self.memory[t]['ht']
            self.grads['o']=-(yt-pt)
            self.grads['V']+=(self.grads['o']).dot(ht.T)
            self.grads['c']+=self.grads['o']
            self.grads['h']=np.dot(self.params['V'].T,self.grads['o']) +self.grads['hpp']
            self.grads['a']=np.multiply(self.grads['h'],(1-np.square(self.memory[t]['ht'])))
            self.grads['U']+= self.grads['a'].dot(xt.T)
            if(t==0):
                hMem=np.zeros_like(self.prevH)
            else:
                hMem=self.memory[t-1]['ht']
            self.grads['W']+=self.grads['a'].dot(hMem.T)
            self.grads['b']+=self.grads['a']
            self.grads['hpp']=(self.params['W'].T).dot(self.grads['a'])

        for grad in self.grads:
            self.grads[grad] = np.clip(self.grads[grad], -5, 5)
            
    def train(self,inputs,targets):

        iters = 0
        max_epochs=5
        smooth_loss=self.calculateLoss( X[0:self.seq_length], Y[0:self.seq_length],self.prevH)
        for epoch in tqdm(range(max_epochs)):
            print()
            curr_iter = 0 
            e = 0 
            self.prevH=np.zeros((self.m,1))
            while e + self.seq_length + 1 < len(inputs):
                start = e 
                end = start + self.seq_length

                X_batch = X[start:end]
                Y_batch = Y[start:end]
 
              


                self.forward(X_batch)

                self.backward(X_batch,Y_batch)

                checkGradients=False
                if(checkGradients):
                    if(epoch==0 and e==0):
                        bak=np.copy(self.params)
                        gradsNum=self.compute_gradients_num(X_batch,Y_batch)
                        np.params=np.copy(bak)
                        for key in gradsNum:
                            print('Max Difference between Gradients for ',key,' is:',np.max(self.grads[key]-gradsNum[key]))


                self.adaGrad()

                loss=self.calculateLoss(X_batch,Y_batch,self.prevH)
                self.trueLoss.append(loss)
                smooth_loss = .999 * smooth_loss + .001 * loss

                self.smoothLoss.append(smooth_loss)

                self.prevH=np.copy(self.memory[-1]['ht'])

                
                if iters % 10000 == 0:
                    logfile='Log_Trump.log'
                    log = open(logfile, "a")
                    bak=sys.stdout
                    sys.stdout = log  
                    print("Epoch: " + str(epoch)+" Total iter: " + str(iters)+"  Local iter: " + str(curr_iter) + "  loss: " + str(loss) +" smooth loss: " + str(smooth_loss))                   
                    print('Text: ',self.getText(self.prevH, X_batch[0], 140))
                    print()
                e += self.seq_length
                iters += 1
                curr_iter += 1
                self.memory=[]
        log.close()
        sys.stdout = bak

    def compute_gradients_num(self, inputs, targets, h=1e-6):
        num_grads  = {"W": np.zeros_like(self.params['W']), "U": np.zeros_like(self.params['U']),
                      "V": np.zeros_like(self.params['V']), "b": np.zeros_like(self.params['b']),
                      "c": np.zeros_like(self.params['c']),
                      }
        for key in ['U','W','V','b','c']:
            for i in range(self.params[key].shape[0]):
                for j in range(self.params[key].shape[1]):
                    old_par = self.params[key][i][j] 
                    self.params[key][i][j] = old_par + h
                    l1=self.calculateLoss(inputs,targets,self.prevH)
                    self.params[key][i][j] = old_par - h
                    l2=self.calculateLoss(inputs,targets,self.prevH)
                    self.params[key][i][j] = old_par 
                    num_grads[key][i][j] = (l1 - l2) / (2*h)
        return num_grads

    def getText(self, h, ix, n):
        xnext = np.zeros((self.K, 1))
        xnext[ix] = 1 
        txt = ''
        for t in range(n):
            at=self.params['W'].dot(h)+self.params['U'].dot(xnext)+self.params['b']
            h=np.tanh(at)
            ot=self.params['V'].dot(h)+self.params['c']
            p=softmax(ot)
            ix = np.random.choice(range(self.K), p=p.flatten())
            xnext = np.zeros((self.K, 1))
            xnext[ix] = 1 
            txt += data['ind_to_char'][ix]
        return txt
        
    def calculateLoss(self,inputs,targets,h):
        n=len(inputs)
        prevH=h
        l=0
        for t in range(n):
            x=np.zeros((self.K,1))
            x[inputs[t]]=1
            at=self.params['W'].dot(prevH)+self.params['U'].dot(x)+self.params['b']
            ht=np.tanh(at)
            ot=self.params['V'].dot(ht)+self.params['c']
            pt=softmax(ot)
            prevH=np.copy(ht)
            l+= -np.log(pt[targets[t]][0])
        return l


def load_data(data):

    characters = list(set(data))
    characters=''.join(sorted(characters))
    characters=list(characters)
    char_dictionary = dict([ (elem, i) for i, elem in enumerate(characters) ])
    inv_char_dictionary = {v: k for k, v in char_dictionary.items()}
    voc_size = len(char_dictionary)

    data = {"data": data, 
            "chars": characters,
            "vocab_len": voc_size,
            "char_to_ind": char_dictionary,
            "ind_to_char": inv_char_dictionary}
    return data


import json

# data=[]
# with open('condensed_2018.json') as f:
#   temp = json.load(f)
#   data=temp
# file=['2017','2016','2015','2014','2013','2012','2011','2010','2009']
# for f in file:
#     filename='condensed_'+f+'.json'
#     with open(filename) as f:
#         temp=(json.load(f))
#     data+=temp

data=[]
with open('test.json', encoding="utf8") as f:
  temp = json.load(f)
  data=temp

text="".join(d['text'] for d in data)
encoded_string = text.encode("ascii", "ignore")
decode_string = encoded_string.decode()
data=load_data(decode_string)




K=data['vocab_len']  # dimensionality of the output (input) vector of your RNN
m=100  # dimensionality of hidden state
eta=0.1
seq_length=25
maxEpochs=2
iter=0
epoch=0
e=0
n=0
rnn=RNN()

X=[data['char_to_ind'][char] for char in data['data'][:len(data['data'])-2]]
Y=[data['char_to_ind'][char] for char in data['data'][1:len(data['data'])-1]]
# X=[data['char_to_ind'][char] for char in data['book_data'][:25]]
# Y=[data['char_to_ind'][char] for char in data['book_data'][1:26]]
rnn.train(X,Y)
rnn.getText(rnn.prevH,data['char_to_ind']['h'] , 140)





