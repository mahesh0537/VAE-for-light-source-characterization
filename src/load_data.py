import pandas as pd
import tensorflow as tf
import numpy as np
import argparse
from utils import str2bool
from params import param




class load_data(object):
    def __init__(self, args):
        self.args = args
        self.genTrainingData = True
        self.list_nbar = []
        self.probs = []
        self.bins = []
        self.lable = []
        self.nbars = []
        self.nbars_SPACS = []
        self.nbars_SPATS = []
        self.nbars_C = []
        self.nbars_T = []
        self.len_data = 0

    def gen_path(self, source = 'SPAC', bin_size = 10):
        if self.genTrainingData:
            source = source + '_nbarTh_' + str(self.args.nbar) + '_QE_' + str(self.args.quantum_efficiency) + '_nSensor_' + str(self.args.n_sensor)
            return self.args.path + '/' + str(source) + str(bin_size) + '.txt'
        else:
            return self.args.path + '/' + str(source) + str(bin_size) + '.txt'
        
    def appendData(self, filename):
        print(filename)
        data = pd.read_csv(filename, sep= " ", header=None)
        if filename.find('SPAC') != -1:
            self.nbars_SPACS.append(np.average(data.iloc[:, 7]))
            label = np.zeros((data.shape[0], 1), dtype=float)
        elif filename.find('SPAT') != -1:
            self.nbars_SPATS.append(np.average(data.iloc[:, 7]))
            label = np.ones((data.shape[0], 1), dtype=float)
        elif filename.find('coherent') != -1:
            self.nbars_C.append(np.average(data.iloc[:, 7]))
            label = np.ones((data.shape[0], 1), dtype=float)*2
        elif filename.find('thermal') != -1:
            self.nbars_T.append(np.average(data.iloc[:, 7]))
            label = np.ones((data.shape[0], 1), dtype=float)*3
        data = pd.concat([data, pd.DataFrame(label)], axis=1)
        self.nbars.append(np.average(data.iloc[:, 7]))
        self.probs.append(data.iloc[:, 0:8].values)
        self.lable.append(data.iloc[:, -1].values)

    def check_nbar_obs(self, filename):
        data = pd.read_csv(filename, sep= " ", header=None)
        return np.average(data.iloc[:, 8:])

    def load(self, bin_size = 200, filenName = None):
        if filenName is None:
            try:
                dataSPACS = pd.read_csv(self.gen_path('SPAC', bin_size), sep= " ", header=None)
                dataSPATS = pd.read_csv(self.gen_path('SPAT', bin_size), sep= " ", header=None)
            except:
                print(self.args)
                raise Exception("File not found")
        else:
            dataSPACS = pd.read_csv(filenName, sep= " ", header=None)
            dataSPATS = pd.read_csv(filenName, sep= " ", header=None)
        dataset = pd.concat([dataSPACS, dataSPATS])

        sourceC = pd.DataFrame( np.zeros((1000,1), dtype=float))
        sourceT = pd.DataFrame( np.ones((1000,1), dtype=float))
        datasetLable = pd.concat([sourceC, sourceT])

        data = [dataset, datasetLable]
        data = pd.concat(data, axis= 1)

        self.nbar=  np.average(data.iloc[:, 7])
        self.list_nbar.append(self.nbar)
        self.nbar_SPACS = np.average(dataSPACS.iloc[:, 7])
        self.nbar_SPATS = np.average(dataSPATS.iloc[:, 7])

        self.nbars_SPATS.append(self.nbar_SPATS)
        self.nbars_SPACS.append(self.nbar_SPACS)
        self.nbars.append(self.nbar)


        if self.args.shuffle:
            data = data.sample(frac = 1)

        self.lable.append(data.iloc[:, -1].values)
        if self.args.probs:
            self.probs.append(data.iloc[:, 0:8].values)
        if self.args.bins:
            self.bins.append(data.iloc[:, 8:-2].values)

    def __call__(self, **kwargs):
        n_nodes = self.args.input_nodes
        # self.load(bin_size)
        # print(self.__str__())
        self.probs = np.concatenate(self.probs, axis = 0)
        self.lable = np.concatenate(self.lable, axis = 0)
        self.len_data = self.probs.shape[0]
        # print(self.probs.shape)
        # print(self.lable.shape)
        if self.args.train:
            print('[INFO] Training data')
            train_size = int(self.args.split_ratio*self.probs.shape[0])
            X_train = tf.cast(np.concatenate((self.probs[:train_size, :n_nodes], np.reshape(self.probs[:train_size, -1], (train_size, 1))), axis = -1), dtype=tf.float32)
            y_train = self.lable[:train_size]
            X_test = tf.cast(np.concatenate((self.probs[train_size:, :n_nodes], np.reshape(self.probs[train_size:, -1], (self.probs.shape[0] - train_size, 1))), axis = -1), dtype=tf.float32)
            y_test = self.lable[train_size:]
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train,y_train))
            train_dataset = train_dataset.shuffle(buffer_size=2048).batch(self.args.batch_size)
            return train_dataset, [X_test, y_test]
        else:
            return tf.cast(np.concatenate((self.probs[:, :n_nodes], np.reshape(self.probs[:, -1], (self.probs.shape[0], 1))), axis = -1), dtype=tf.float32), self.lable


    def __str__(self) -> str:
        out =  f"nbar: {np.round(np.average(self.nbars), 2)}"
        if len(self.nbars_SPACS) > 0:
            out += f", nbar_SPACS: {np.round(np.average(self.nbars_SPACS), 2)}"
        if len(self.nbars_SPATS) > 0:
            out += f", nbar_SPATS: {np.round(np.average(self.nbars_SPATS), 2)}"
        if len(self.nbars_C) > 0:
            out += f", nbar_C: {np.round(np.average(self.nbars_C), 2)}"
        if len(self.nbars_T) > 0:
            out += f", nbar_T: {np.round(np.average(self.nbars_T), 2)}"
        out += f", len_data: {self.len_data}"
        return out


if __name__ == "__main__":
    import os
    args = param()()


    # name = gen_name(args)
    # print(name)
    path = "../train_data_nbarObs_1.3"
    print(path)
    args.path = path

    data_loder = load_data(args)
    
    filenames = os.listdir(path)
    for filename in filenames:
        data_loder.appendData(path + '/' + filename)
    data = data_loder()
    print(data_loder)
    print(data)
    classes = np.unique(data[1])
    print([np.sum(data[1] == c) for c in classes])