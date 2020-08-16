import numpy as np
from normalize import RegressionNormalize, NoNormalize
from theta_generator import ThetaGenerator

class SimpleLinearRegression(object):
    def __init__(self, dataset, is_normalized, theta_generator):
        self.__dataset = np.copy(dataset) #data set for training
        self.__normalize = RegressionNormalize() if is_normalized else NoNormalize() #normalize 
        self.__theta_generator = theta_generator # theta generator
        self.__theta = None #theta vector

    @property
    def theta_generator(self):
        return self.__theta_generator

    @theta_generator.setter
    def theta_generator(self, generator):
        self.__theta_generator = generator

    @property
    def theta(self):
        return self.__theta


    def __setupModels(self):
        '''
            get input and output pair for learning
            @return (input set, output set)
        '''
        X = np.zeros((np.size(self.__dataset, 0), np.size(self.__dataset, 1)))
        X[:, 0] = 1
        X[:,1:] = self.__dataset[:,:-1]
        Y = self.__dataset[:,-1]

        return X, Y

    @property
    def input_set(self):
        return self.__setupModels()[0]

    @property
    def output_set(self):
        return self.__setupModels()[1]

    def learn(self):
        '''
            Generate Theta vector
        '''
        X, Y = self.__setupModels()
        self.__theta = self.__theta_generator.generate(self.__normalize.normalize_input(X), Y)
    
    def learn_more(self, dataset):
        '''
            Append new data sets and learn again
            @param dataset appended dataset
        '''
        self.__dataset = np.append(self.__dataset, dataset, axis=0)
        self.learn()

    def predict(self, predict_set):
        '''
            Predict the set
            Note: Each row of predict_set must be start with 1
            @param predict_set set that need to be predicted
            @Exception when it have not learnt before invoke this method
            @return predicted set
        '''
        if self.__theta is None:
            raise Exception('It needs to be learnt before predicting.')
       
        return ThetaGenerator.predict(self.__normalize.normalize_predict(predict_set), self.__theta)    
