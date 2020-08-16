import numpy as np

class Normalize(object):
    def normalize_input(self, X):
        '''
            Get nomalized input set
            @param X input set
            @return normalized input set
        '''
        pass

    def normalize_predict(self, predict_set):
        '''
            Get normalized set of set needs to be predicted
            @param predict_set set that needs to be predicted
            @return normalized set of set needs to be predicted
        '''
        pass

class RegressionNormalize(Normalize):
    def __init__(self):
        self.__mu = 0.0
        self.__standard = 0.0

    @property
    def mu(self):
        return self.__mu
    
    @property
    def standard(self):
        return self.__standard


    def normalize_input(self, X):
        normal = np.copy(X)
        normal[0,0] = 100
        
        self.__standard = np.std(normal, 0, dtype=np.float64)
        self.__mu = np.mean(normal, 0)
        
        normal = (normal - self.__mu) / self.__standard
        normal[:,0] = 1

        return normal

    def normalize_predict(self, predict_set):
        normal = (np.copy(predict_set) - self.__mu) / self.__standard
        normal[0] = 1
        return normal

class NoNormalize(Normalize):
    def normalize_input(self, X):
        return X

    def normalize_predict(self, predict_set):
        return predict_set
