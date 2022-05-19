import numpy as np
from matplotlib import pyplot


class Model:
    def __init__(self, N):
        self.w0 = np.random.uniform(-0.1, 0.1, (N, 1))
        self.b0 = np.random.uniform(-0.1, 0.1, (N, 1))
        self.act0 = np.tanh

        self.w1 = np.random.uniform(-0.1, 0.1, (1, N))
        self.b1 = np.random.uniform(-0.1, 0.1, (1, 1))

    def forward(self, x ):
        h0_ = np.dot(self.w0, x) + self.b0
        h0 = np.tanh(h0_)

        h1 = np.dot(self.w1, h0) + self.b1

        return (h0_, h0, h1)

    def backward(self, x, d, lr):
        (h0_, h0, h1) = self.forward(x)
        grad = (h1 - d)
        self.w1 -= lr * np.dot(grad, h1.T)

        self.b1 -= lr * grad

        grad = np.dot(self.w1.T, grad) * (np.ones(h0_.shape) - np.tanh(h0_)**2)

        self.w0 -= lr * np.dot(grad, x.T)
        self.b0 -= lr * grad
        return None


def train(x,d,epoch,lr,model):
    (_, _, pred) = model.forward(x)
    MSE_LOSS = (np.square(d - pred)).mean(axis=1)
    MSE = [MSE_LOSS]
    
    for i in range(1,epoch):
        (_, _, pred) = model.forward(x)
        MSE_LOSS = (np.square(d - pred)).mean(axis=1)
        print('epoch {} : MSE_LOSS = {}'.format(i, MSE_LOSS))
        for j in range(x.shape[1]):
            model.backward(x[0, j], d[0, j], lr)
        MSE.append(MSE_LOSS)
        if MSE[i] > MSE[i-1]:
            lr *= 0.998
        else:
            lr *= 1.000001

    return MSE


if __name__ == '__main__':
    bp = Model(24)
    n = 300
    N = 24
    x = np.random.uniform(0, 1, (1, n))
    v = np.random.uniform(-0.1, 0.1, (1, n))
    d = np.sin(20 * x) + 3 * x + v

    model = Model(N)
    epoch = 1000000
    lr = 0.005
    MSE = train(x,d,epoch,lr,model)
    x = np.arange(0, 1, 0.001).reshape(1000,1)

    v = np.random.uniform(-0.1, 0.1, (1000, 1))
    d = np.sin(20 * x) + 3 * x + v

    (_, _, y) = model.forward(x.reshape(1,1000))
    y = y.reshape(1,1000)




    '''way to print figur
    fig1 = pyplot.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.scatter(x, d, label='d')
    ax1.scatter(x, y, label='y')
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_xlabel('Input')
    ax1.set_ylabel('Output')
    ax1.set_title('curve fitting')

    fig2 = pyplot.figure()
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.plot(range(epoch), MSE)
    ax2.set_xlim(1, epoch)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE')
    ax2.set_title('mse loss')

    pyplot.show()
    '''