import random
import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import pickle
    
DIM=28

def read_image(image_path, label_path):
    X=[] 
    for i,path in enumerate(image_path):
        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img,(DIM,DIM))
        X.append(img)
        print('reading {}/{}'.format(i+1,len(image_path)),end='\r')
    print("")

    X=np.array(X)
    df = pd.read_csv(label_path)
    y = df[df['filename'].isin([path.split(sep=os.sep)[-1] for path in image_path])]['digit'].values
    yvector = np.zeros((len(y), 10))
    for i in range(len(y)):
        yvector[i, y[i]] = 1
    return X, yvector
        

class modelpart:
    def __init__(self):
        pass

class convolution(modelpart):

    def __init__(self, kernel_size, stride, padding, rate):
        self.Ksize = kernel_size
        self.stride = stride
        self.padding = padding
        self.lr = rate
        self.W = np.zeros((self.Ksize, self.Ksize))
        self.B = 0

        for i in range(self.Ksize):
            for j in range(self.Ksize):
                self.W[i, j] = random.uniform(-0.1, 0.1)
                    


    def convart(self, delY, W):
        # dilate delY by self.stride - 1
        n, m, inchannel = self.Xshape
        output_size = (n + 2 * self.padding - self.Ksize) // self.stride + 1
        tmpdelY = np.zeros((output_size + (output_size - 1) * (self.stride - 1), output_size + (output_size - 1) * (self.stride - 1)))
        tmpW = np.copy(W)
        tmpW[:,:] = np.rot90(tmpW[:,:], 2)
        return tmpdelY, tmpW 

    def forward(self, X):
        # print(X.shape)
        n, m, inchannel = X.shape
        assert n == m, "image is not square"

        self.Xshape = X.shape

        output_size = (n + 2 * self.padding - self.Ksize) // self.stride + 1
        tmpX = np.zeros((n + 2 * self.padding, n + 2 * self.padding))
        for i in range(n):
            for j in range(n):
                tmpX[i + self.padding, j + self.padding] = X[i, j, 0]

        self.padedX = tmpX
        n += 2 * self.padding
        output = np.zeros((output_size, output_size, 1))

        for i,i1 in zip(range(0, n, self.stride), range(output_size)):
            for j,j1 in zip(range(0, n, self.stride), range(output_size)):
                output[i1, j1, 0] += np.sum(tmpX[i:i + self.Ksize, j:j + self.Ksize] * self.W) + self.B

        # print("\nconv forward done\n")

        return output


    def backward(self, delY):
        n, m, inchannel = self.Xshape

        delX = np.zeros(self.Xshape)
        delW = np.zeros(self.W.shape)
        delB = 0
        
        delB = np.sum(delY)

        Y2 , W2 = self.convart(delY, self.W)
        N, M = Y2.shape
        Y3 = np.copy(Y2)

        Y2 = np.zeros((N + self.Ksize - 1, N + self.Ksize - 1))
        for i in range(N):
            for j in range(M):
                Y2[i + self.Ksize - 1, j + self.Ksize - 1] = Y3[i, j]
        
        for i in range(n):
            for j in range(n):
                delX[i,j,0] += np.sum(Y2[i:i+self.Ksize, j:j+self.Ksize] * W2[:,:]) 
        for i in range(self.Ksize):
            for j in range(self.Ksize):
                delW[i,j] = np.sum(Y3[:,:] * self.padedX[i:i+N,j:j+N])
        

        self.W -= self.lr * delW
        self.B -= self.lr * delB
        return delX


class relu(modelpart):
    def __init__(self):
        self.X = None

    def forward(self, X):

        self.X = np.copy(X)
        X = np.maximum(X, 0)
        
        return X

    def backward(self, delY):
        delX = np.copy(delY)
        delX[self.X < 0] = 0
        return delX


class maxpooling(modelpart):
    def __init__(self, size, stride):
        self.n = size
        self.stride = stride

    def forward(self, X):
        self.X = X

        # print(X.shape, " is X shape forward in pooling")
        N, M, inchannel = X.shape
        assert N==M, "image is not square"
        N2 = (N - self.n)//self.stride + 1
        output = np.zeros((N2, N2, 1))

        # plt.clf()
        # plt.imshow(X[:,:,0], cmap=plt.get_cmap('gray'))
        # plt.plot()
        # plt.show()


        for i, i1 in zip(range(0, N, self.stride), range(N2)):
            for j, j1 in zip(range(0, N, self.stride), range(N2)):
                output[i1, j1, 0] = np.max(X[i:i + self.n, j:j + self.n, 0])

        # plt.clf()
        # plt.imshow(output[:,:,0], cmap=plt.get_cmap('gray'))
        # plt.plot()
        # plt.show()

        return output

    def backward(self, delY):
        N, M, inchannel = self.X.shape
        delX = np.zeros(self.X.shape)
        N2 = (N - self.n)//self.stride + 1
        
        for i, i1 in zip(range(0, N, self.stride), range(N2)):
            for j, j1 in zip(range(0, N, self.stride), range(N2)):
                pos = np.argmax(self.X[i:i + self.n, j:j + self.n, 0])
                (idx, idy) = np.unravel_index(pos, (self.n, self.n))
                delX[i+idx, j+idy, 0] = delY[i1, j1, 0]
        return delX


class flatten(modelpart):
    def __init__(self):
        pass

    def forward(self, inputs):
        # print("forward flatten")
        self.W, self.H, fau = inputs.shape
        # print(inputs.shape, "flatten forward shape")
        return inputs.reshape(self.W*self.H)
    def backward(self, delY):
        return delY.reshape(self.W, self.H, 1)


class fullyconnected(modelpart):    
    def __init__(self, N, M, rate):
        self.W = np.random.randn(N, M) * 0.01
        self.B = np.zeros(M)
        self.lr = rate
        
    def forward(self, X):
        self.X = X
        return np.dot(self.X, self.W) + self.B

    def backward(self, delY):

        delX = delY @ self.W.T
        delW = np.einsum('i,j->ij', self.X.T, delY)
        delB = delY

        self.W -= self.lr * delW
        self.B -= self.lr * delB

        return delX


class softmax(modelpart):
    def __init__(self):
        pass

    def forward(self, X):
        ex = np.exp(X)
        self.output = ex / np.sum(ex)
        return self.output

    def backward(self, delY):
        return delY


class CNN:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return np.argmax(X)

    def train(self, X, y):
        for layer in self.layers:
            X = layer.forward(X)
        
        delV = X - y
        # print(delV)
        for i in range(len(self.layers)-1, -1, -1):
            delV = self.layers[i].backward(delV)

        # print("training done ")
        

if __name__ == "__main__":

    rate = 0.0001

    cnn = CNN()

    cnn.add(convolution(kernel_size=5, stride=1, padding=2, rate=rate))
    cnn.add(relu())
    # cnn.add(convolution(kernel_size=5, stride=1, padding=2, rate=rate))
    # cnn.add(maxpooling(size=2, stride=2))
    # cnn.add(relu())
    cnn.add(maxpooling(size=2, stride=2))
    cnn.add(flatten())
    cnn.add(fullyconnected(N=14*14, M=120, rate=rate))
    cnn.add(relu())
    cnn.add(fullyconnected(N=120, M=84, rate=rate))
    cnn.add(relu())
    cnn.add(fullyconnected(N=84, M=10, rate=rate))
    cnn.add(softmax())

    a_path = glob.glob("../training-a2"+"/"+"*.png")
    b_path = glob.glob("../training-b"+"/"+"*.png")
    c_path = glob.glob("../training-c2"+"/"+"*.png")
    d_path = glob.glob("../training-d2"+"/"+"*.png")
    
    a_label_path = "../training-a.csv"
    b_label_path = "../training-b.csv"
    c_label_path = "../training-c.csv"
    d_label_path = "../training-d.csv"

    dataA, labelA = read_image(a_path, a_label_path)
    dataB, labelB = read_image(b_path, b_label_path)
    dataC, labelC = read_image(c_path, c_label_path)
    dataD, labelD = read_image(d_path, d_label_path)

    X_train_all = np.concatenate((dataA, dataB, dataC), axis=0)
    y_train_all = np.concatenate((labelA, labelB, labelC), axis=0)
    X_test_all = dataD
    y_test_all = labelD

    X_train_all = X_train_all.reshape(X_train_all.shape[0],DIM, DIM,1).astype('float32')
    X_test_all = X_test_all.reshape(X_test_all.shape[0],DIM, DIM,1).astype('float32')

    X_train_all = X_train_all/255
    X_test_all=X_test_all/255

    indices=list(range(len(X_train_all)))
    np.random.seed(5253)
    np.random.shuffle(indices)

    ind=int(len(indices)*0.90)
    # train data
    X_train=X_train_all[indices[:ind]] 
    y_train=y_train_all[indices[:ind]]

    ind = 60
    X_test_all=X_test_all[:ind]
    y_test_all=y_test_all[:ind]

    sz = len(X_train)
    print(sz, " = size of X_train")

    for i in range(sz):
        # if np.argmax(y_train[i])>=7:
        cnn.train(X_train[i], y_train[i])
        print('training {}/{}'.format(i+1,sz),end='\r')

    print("")
    print("train done")

    sz = X_test_all.shape[0]
    matches = 0
    print("starting test")
    for i in range(sz):
        j = cnn.predict(X_test_all[i])
        actual = np.argmax(abs(y_test_all[i]))
        print("actual = ", actual, " predicted = ", j)
        # end = '\r'
        # if(i==sz-1):
        #     end = '\n'
        # print('predicting {}/{}'.format(i+1,sz),end='\r')
        if(j==actual):
            matches += 1

    print(matches/sz*100, " is the accuracy")
    print("total tested = ", sz)

    pickle.dump(cnn, open("1705094_model.pkl", "wb"))