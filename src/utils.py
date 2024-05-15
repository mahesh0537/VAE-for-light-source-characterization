import argparse
import numpy as np
from copy import deepcopy
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def gen_name(args):
    return "_{}_{}_{}_{}".format(args.involve_loss, args.n_sensor, args.max_photons, args.mixRatio)

class variables:
    def __init__(self, power):
        self.power = power


class term:
    def __init__(self, coff, name = ''):
        self.coff = coff
        self.list_atr = []
        self.power = 0
        self.name = name
    
    def add_var(self, var, val):
        if var in self.list_atr:
            getattr(self, str(var)).power += val.power
            self.power += val.power
        else:
            self.list_atr.append(var)
            self.power += val.power
            setattr(self, str(var), val)
            
    def get_val(self, var):
        return getattr(self, str(var))

    def update_power(self):
        temp = 0
        for i in self.list_atr:
            temp += getattr(self, str(i)).power
        return temp
    
    def format(self, id):
        return "{}^{}*".format(id, self.get_val(id).power)
    
    def __str__(self):
        temp = str(self.coff)+"*"
        ids = np.sort(self.list_atr)
        for id in ids:
            temp += self.format(id)
        return temp[:-1]

    def _vars(self):
        temp = ""
        ids = np.sort(self.list_atr)
        for id in ids:
            temp += self.format(id)
        return temp

    def __mul__(self, other):
        temp = term(self.coff*other.coff)
        for i in self.list_atr:
            temp.add_var(i, deepcopy(self.get_val(i)))
        for i in other.list_atr:
            temp.add_var(i, deepcopy(other.get_val(i)))
        return temp

    def __add__(self, other):
        if (self._vars() == other._vars()):
            temp = term(self.coff + other.coff)
            for i in self.list_atr:
                temp.add_var(i, deepcopy(self.get_val(i)))
            return temp
        else:
            print('error')

    def __sub__(self, other):
        if (self._vars() == other._vars()):
            temp = term(self.coff - other.coff)
            for i in self.list_atr:
                temp.add_var(i, deepcopy(self.get_val(i)))
            return temp
        else:
            print('error')

    def __repr__(self):
        return self.__str__()

class multinomial:
    def __init__(self, terms):
        self.terms = terms
    def __mul__(self, other):
        temp = []
        for i in self.terms:
            for j in other.terms:
                temp.append(i*j)
        return multinomial(temp).sort()

    def sort(self):
        temp = []
        temp_terms = []
        for i in range(0, len(self.terms)):
            if self.terms[i]._vars()  in temp:
                idx = np.argwhere(np.array(temp) == self.terms[i]._vars())[0, 0]
                temp_terms[idx] += self.terms[i]
            else:
                temp.append(self.terms[i]._vars())
                temp_terms.append(self.terms[i])
        return multinomial(temp_terms)

    def __str__(self):
        temp = ""
        for i in self.terms:
            temp += (" + " + str(i))
        return temp[3:]
    
    def out(self, n):
        temp = 0
        for i in self.terms:
            if len(i.list_atr) == n:
                temp += i.coff
        return temp
    
    def total_coff(self):
        temp = 0
        for i in self.terms:
            temp += i.coff
        return temp

class sensor:
    def __init__(self, n_sensor):
        alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        self.n_sensor = n_sensor
        self.var = [str(alphabet[i]) for i in range(0, n_sensor)]
        self.list_terms = [term(1, i) for i in self.var]
        for i in self.list_terms:
            i.add_var(i.name, variables(1))
        self.seed = multinomial(self.list_terms)
    
    def calc(self, n_photon):
        temp = deepcopy(self.seed)
        for i in range(1, n_photon):
            temp *= self.seed
        return temp
    
    def __call__(self, probs):
        out_prob = np.zeros(self.n_sensor + 1)
        out_prob[0] = probs[0]
        for i in range(1, len(probs)):
            coffs = self.calc(i)
            temp_sum = coffs.total_coff()
            for j in range(1, self.n_sensor + 1):
                out_prob[j] += coffs.out(j)*probs[i]/temp_sum
        return out_prob
