import numpy as np

class ThetaGenerator(object):
    @staticmethod
    def predict(predict_set, Theta):
        '''
            Predict the set
            @param predict_set set needs to be predicted
            @param theta theta vector
            @return predicted set
        '''
        return predict_set @ Theta
    @staticmethod
    def compute_cost(X, Y, Theta):
        '''
            compute the cost of using theta as the paramerter for linear regression to fit the data point in x and y
            @param X input set
            @param Y output set
            @param Theta theta vector
            @retrun the cost 
        '''
        predicted = ThetaGenerator.predict(X, Theta)

        #square error
        sqr_error = (predicted - Y) ** 2

        #sum of square errors
        sum_errors = np.sum(sqr_error)

        return (1 / (2 * np.size(Y)) * sum_errors)
    
    @staticmethod
    def compute_cost_vec(X, Y, Theta):
        '''
            compute the cost of using theta as the paramerter for linear regression to fit the data in each (x, y) pair
            @param X input set
            @param Y output set
            @param Theta theta vector
            @return the cost of each pair
        '''
        error = ThetaGenerator.predict(X, Theta) - Y

        return (1 / (2 * np.size(Y)) * np.transpose(error)@error)

    def generate(self, X, Y):
        '''
            Generate Theta vector
            @param X input set
            @param Y output set
            @return Theta vector
        '''
        pass

class GradientDescent(ThetaGenerator):
    def __init__(self, alpha, iterate_num):
        self.__alpha = alpha
        self.__iterate_num = iterate_num
        self.__cost_history = None

    @property
    def alpha(self):
        return self.__alpha

    @alpha.setter
    def alpha(self, alpha):
        self.__alpha = alpha

    @property
    def iterate_num(self):
        return self.__iterate_num

    @iterate_num.setter
    def iterate_num(self, iterate_num):
        self.__iterate_num = iterate_num

    @property
    def cost_history(self):
        return self.__cost_history

    def generate(self, X, Y):
        theta = np.zeros(np.size(X, 1))
        self.__cost_history = np.zeros((self.__iterate_num, 2))
        m = np.size(Y)
        X_T = np.transpose(X)
        pre_cost = ThetaGenerator.compute_cost(X, Y, theta)

        for i in range(0, self.__iterate_num):
            error = ThetaGenerator.predict(X, theta) - Y
            theta = theta - (self.__alpha / m) * (X_T @ error)
            cost = ThetaGenerator.compute_cost(X, Y, theta)
            #check if theta is good enough
            if np.round(cost, 15) == np.round(pre_cost, 15):
                self.__cost_history[i:,0] = range(i, self.__iterate_num)
                self.__cost_history[i:,1] = cost
                break

            pre_cost = cost
            self.__cost_history[i, 0] = i
            self.__cost_history[i, 1] = cost

        return theta


class NormalEquation(ThetaGenerator):
    def generate(self, X, Y):
        return np.linalg.pinv(X.T @ X) @ (X.T @ Y)
