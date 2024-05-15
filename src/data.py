import numpy as np
import matplotlib.pyplot as plt
import math
from copy import deepcopy
from math import comb, factorial
import matplotlib.pyplot as plt
import os
from utils import sensor
from params import param


class data(object):
    def __init__(self, args):
        self.genTrainingData = True
        self.args = args
        self.nbar = args.nbar
        self.nData_points = args.nData_points
        self.num_sensors = args.n_sensor
        self.quantum_efficiency = args.quantum_efficiency
        self.path = args.path
        self.max_photons = args.max_photons
        if self.args.involve_loss:
            self.nbar_b = (args.nbar - 1)/2
        else:
            self.nbar_b = (args.nbar - 1)/2   #nbar before single photon
        self.printLog = False


    def updateFileName(self):
        try:
            self.source = self.source + '_nbarTh_' + str(int(self.nbar * 10)/10.0) + '_QE_' + str(self.quantum_efficiency) + '_nSensor_' + str(self.num_sensors) + '_mixRatio_' + str(self.mixRatio)
            if self.mixRatio == 1.0:
                if 'SPAT' in self.source:
                    self.source = 'thermal'
                elif 'SPAC' in self.source:
                    self.source = 'coherent'
        except:
            self.source = self.source + '_nbarTh_' + str(int(self.nbar * 10)/10.0) + '_QE_' + str(self.quantum_efficiency) + '_nSensor_' + str(self.num_sensors)



    def prob_calc(self, n):
        return NotImplementedError

    
    def involve_sensor(self, prob_last_state = None):
        if prob_last_state is None:
            prob_last_state = self.prob_th
        sim_sensor = sensor(self.num_sensors)
        self.prob_sensor = sim_sensor(prob_last_state)


    def involve_quantum_efficiency(self, prob_last_state = None):
        if prob_last_state is None:
            prob_last_state = self.prob_th
        prob = [0 for i in range(0, len(prob_last_state))]
        for n in range(0, len(prob_last_state)):
            for m in range(n, len(prob_last_state)):
                prob[n] += comb(m, n)*(self.quantum_efficiency**n)*((1-self.quantum_efficiency)**(m-n))*prob_last_state[m]
        self.prob_quantum_loss = np.array(prob)

    def calc(self, loss = True):
        self.prob = []
        NExperiment = 1e8
        NExperimentFinal = NExperiment
        for i in range (0, self.max_photons+1):
            self.prob.append([i, self.prob_calc(i)])
            print(f'[INFO] i: {i}, prob: {self.prob[-1][1]}')
            
        self.prob = np.array(self.prob)
        print(f'[INFO] prob: {self.prob}')
        self.prob_th = self.prob[:, 1]
        self.prob_final = self.prob_th
        if self.printLog:
            print(self.prob_th, ' ', 1 - np.sum(self.prob_th), 'prob_th')
        self.p = (self.prob_final*(NExperiment)).astype('int')
        for i in range(0, self.num_sensors +1):
            if self.printLog:
                print('P without any loss['+str(i)+']',self.p[i])
        if loss == True:
            self.involve_quantum_efficiency(self.prob_th)
            self.involve_sensor(self.prob_quantum_loss)
            self.prob_final = self.prob_sensor
        ##
        self.p = (self.prob_final*(NExperiment)).astype('int')
        if self.printLog:
            for i in range(0, self.num_sensors +1):
                print('P After Noise['+str(i)+']',self.p[i])
        
        
        nPhoton = np.arange(0, self.num_sensors +2)
        if loss:
            wt = self.prob_final
        else:
            wt = self.prob_final[:self.num_sensors +1]
        
        if (1 - np.sum(wt)) < 0:
            wt = wt/np.sum(wt)
            if sum(wt) > 1:
                wt = wt/sum(wt)
        # print(wt, 'wt norm if less than 0', sum(wt))
        if (1 - np.sum(wt)) > 0:
            wt =np.append(wt,1 - np.sum(wt))
            # print(wt, 'wt norm if > than 0', sum(wt))
        elif (1 - np.sum(wt)) == 0:
            wt =np.append(wt,0)
        self.data = np.random.choice(nPhoton, p = wt, size = (int(NExperimentFinal)))
        self.data = np.delete(self.data, np.argwhere(self.data > self.num_sensors))
        if self.printLog:
            print(wt, 'wt', sum(wt))
            print(np.argwhere(self.data > self.num_sensors).shape, 'Photon count modre than sensor count error count')
            print(self.data.max())

    def estimate_nbar(self):
        self.nbarObs = np.sum(self.data)/len(self.data)
        print('Nbar = ' + str(self.nbarObs))
        return self.nbarObs
    
    def save(self, k, bin_size, save = True, target_nbar = None, error_range = 0.01, key = ''):
        self.bin_size = bin_size
        final_save_data = []
        just_data = []
        for i in range(0, int(self.nData_points)):
            temp_i = k + i*bin_size
            temp_p = []
            temp_ = self.data[temp_i:temp_i+self.bin_size]
            
            for i in range(0, 7):
                temp_p.append(temp_[temp_==i].shape[0]/self.bin_size)
            temp_p.append(np.sum(temp_)/self.bin_size)
            temp_arr = np.append(temp_p, temp_)
            final_save_data.append(temp_arr)
            just_data.append(temp_)
        self.calculatedNbar = np.sum(just_data)/(self.bin_size*(len(just_data)))
        print('Calculated Nbar '+self.source+' = ' + str(self.calculatedNbar))
        if target_nbar is None:
            if save:
                np.savetxt(str(self.path)+"/"+str(self.source)+str(key)+str(self.bin_size)+".txt", final_save_data, delimiter=' ', fmt='%s')
            return
        if abs(self.calculatedNbar - target_nbar) < error_range and self.calculatedNbar >= target_nbar:
            if save:
                np.savetxt(str(self.path)+"/"+str(self.source)+str(key)+str(self.bin_size)+".txt", final_save_data, delimiter=' ', fmt='%s')
            return True, self.calculatedNbar
        else:
            return False, self.calculatedNbar
    

    def __call__(self):
        if self.args.involve_loss == False:
            self.num_sensors = 6
        self.calc(self.args.involve_loss)

    

    
    

class SPAC(data):
    def __init__ (self, args):
        super().__init__(args)
        offset = 0.15*self.nbar_b
        self.nbar += offset
        self.nbar_b += offset
        self.source = 'SPAC'
        if self.genTrainingData:
            self.updateFileName()
    
    def prob_calc(self, n):

        # self.nbar_b += 0.1
        a = math.exp(-self.nbar_b)/(1+self.nbar_b)
        atleat_onecase = False
        if n == 1:
            a *= self.nbar_b**(n-1)/math.factorial(n - 1)
            atleat_onecase = True
        elif n >=2:
            a *= (self.nbar_b*(self.nbar_b**(n-2))/math.factorial(n - 2) + self.nbar_b**(n-1)/math.factorial(n - 1))
            atleat_onecase = True
        if not atleat_onecase:
            return 0
        return a


class SPAT(data):
    def __init__ (self, args):
        super().__init__(args)
        self.L = 1
        offset = 0.02*self.nbar_b
        self.nbar += offset
        self.nbar_b += offset
        self.source = 'SPAT'
        if self.genTrainingData:
            self.updateFileName()
    
    def prob_calc(self, n):
        a = math.pow(self.nbar_b, n - self.L)/math.pow(self.nbar_b + 1, n + self.L)
        if n >= self.L:
            a *= math.factorial(n)/(math.factorial(self.L)*math.factorial(n - self.L))
        else:
            return 0
        return a


#####mixed states########
class mixSPAC(data):
    def __init__ (self, args):
        super().__init__(args)
        offset = 0.15*self.nbar_b
        self.nbar += offset
        self.nbar_b += offset
        self.source = 'mixSPAC'
        self.mixRatio = args.mixRatio
        if self.genTrainingData:
            self.updateFileName()
    
    def prob_calc_(self, n, nbar_b = None):
        if nbar_b == None:
            nbar_b = self.nbar_b

        # self.nbar_b += 0.1
        a = math.exp(-nbar_b)/(1+nbar_b)
        atleat_onecase = False
        if n == 1:
            a *= nbar_b**(n-1)/math.factorial(n - 1)
            atleat_onecase = True
        elif n >=2:
            a *= (nbar_b*(nbar_b**(n-2))/math.factorial(n - 2) + nbar_b**(n-1)/math.factorial(n - 1))
            atleat_onecase = True
        if not atleat_onecase:
            return 0
        return a
    
    def Cprob_calc(self, n):
        # return self.prob_calc_(n, nbar_b=self.nbar)
        return ((self.nbar**n)/factorial(n))*math.exp(-self.nbar)

    def prob_calc(self, n):
        return self.mixRatio*self.Cprob_calc(n) + (1.0 - self.mixRatio)*self.prob_calc_(n)



class mixSPAT(data):
    def __init__ (self, args):
        super().__init__(args)
        self.L = 1
        offset = 0.02*self.nbar_b
        self.nbar += offset
        self.nbar_b += offset
        self.source = 'mixSPAT'
        self.mixRatio = args.mixRatio
        if self.genTrainingData:
            self.updateFileName()
    
    def prob_calc_(self, n, nbar_b = None):
        if nbar_b == None:
            nbar_b = self.nbar_b
        a = math.pow(nbar_b, n - self.L)/math.pow(nbar_b + 1, n + self.L)
        if n >= self.L:
            a *= math.factorial(n)/(math.factorial(self.L)*math.factorial(n - self.L))
        else:
            return 0
        return a
    
    def Tprob_calc(self, n):
        return (self.nbar**n)/((1 + self.nbar)**(n + 1))

    def prob_calc(self, n):
        return self.mixRatio*self.Tprob_calc(n) + (1.0 - self.mixRatio)*self.prob_calc_(n)

    