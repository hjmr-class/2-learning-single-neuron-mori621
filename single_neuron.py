import math
import random

def get_teach():
    teach_x = [(0, 0), (0, 1), (1, 0), (1, 1)]
    teach_y = [0, 1, 1, 1]
    return teach_x, teach_y

class SingleNeuron():
    def __init__(self, teach_x, teach_y, alpha=0.01):
        self.teach_x = teach_x
        self.teach_y = teach_y
        self.alpha = alpha

        self.teach_num = len(teach_x)
        self.input_num = len(teach_x[0])

    def sigmoid(self, u):
        return 1.0 / (1.0 + math.exp(-u))

    def forward(self, x):
        u = 0
        for i in range(self.input_num):
            u += self.w[i] * x[i]
        u += self.theta
        return self.sigmoid(u)

    def func_error(self):
        e = 0
        for t in range(self.teach_num):
            y = self.forward(self.teach_x[t])
            e += 0.5 * (y - self.teach_y[t]) * (y - self.teach_y[t])
        return e

    def clear_d(self):
        self.dw = [0 for _ in range(self.input_num)]
        self.dtheta = 0

    def calc_d(self):
        for t in range(self.teach_num):
            y = self.forward(self.teach_x[t])
            for i in range(self.input_num):
                self.dw[i] += (y - self.teach_y[t]) * y * (1 - y) * self.teach_x[t][i]
            self.dtheta += (y - self.teach_y[t]) * y * (1 - y)

    def init_parameter(self):
        self.w = [random.random() - 0.5 for _ in range(self.input_num)]
        self.theta = random.random() - 0.5

    def update_parameter(self):
        self.w = [self.w[i]-self.alpha*self.dw[i] for i in range(self.input_num)]
        self.theta -= self.alpha * self.dtheta


if __name__=='__main__':
    teach_x, teach_y = get_teach()

    model = SingleNeuron(teach_x, teach_y)
    model.init_parameter()
    for i in range(100000):

        if i % 1000 == 0:
            print('{}, {}'.format(i, model.func_error()))
        
        model.clear_d()
        model.calc_d()
        model.update_parameter()

    print('{}, {}'.format(i, model.func_error()))

    for t in range(model.teach_num):
        y = model.forward(teach_x[t])
        print('{}: y={}, y_hat={}'.format(t, y, teach_y[t]))