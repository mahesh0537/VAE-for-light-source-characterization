import argparse
from utils import str2bool, gen_name
import numpy as np


def gen_data_paths():
    paths = {0.8:[1.9, 1.8, 1.7, 1.6],
                          0.6:[1.9, 1.8, 1.7, 1.6]}
                    #   0.4:[1.9, 1.8, 1.7, 1.6]}
    out_paths = []
    for qe in paths.keys():
        for nbar in paths[qe]:
            out_paths.append(f'SPAC_nbarTh_{nbar}_QE_{qe}_nSensor_4200.txt')
            out_paths.append(f'SPAT_nbarTh_{nbar}_QE_{qe}_nSensor_4200.txt')
    return out_paths

class Args:
    def __init__(self):
        pass
    def add_argument(self, attr, type = None, default = None):
        value = None
        if default is not None:
            value = default
        setattr(self, attr[2:], value)
    
    def parse_args(self):
        return self
    
    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)

def update_args(args):
    args.name = gen_name(args)
    args.input_nodes = args.n_sensor + 1
    args.path = "../Data/nbar"+str(args.name)
    return args

class param:
    def __init__(self, ipynb = False):
        if ipynb:
            self.parser = Args()
        else:
            self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--nbar', type=float, default=1.9)
        self.parser.add_argument('--n_sensor', type=int, default=4)
        self.parser.add_argument('--involve_loss', type=str2bool, default=True)
        self.parser.add_argument('--quantum_efficiency', type = float, default=0.8)
        self.parser.add_argument('--max_photons', type=int, default=20)
        self.parser.add_argument('--nData_points', type=int, default=1000)

        self.parser.add_argument('--data_path', type=str, default='')
        self.parser.add_argument('--shuffle', type=str2bool, default=True)
        self.parser.add_argument('--probs', type=str2bool, default=True)
        self.parser.add_argument('--bins', type=str2bool, default=False)
        self.parser.add_argument('--train', type=str2bool, default=False)
        self.parser.add_argument('--split_ratio', type=float, default=0.99999)
        self.parser.add_argument('--input_nodes', type=int, default=3)

        self.parser.add_argument('--epochs', type=int, default=10000)
        self.parser.add_argument('--batch_size', type=int, default=512)
        self.parser.add_argument('--lr', type=float, default=0.001)
        self.parser.add_argument('--out_dir', type=str, default='./checkpoints_new')
        self.parser.add_argument('--cl_loss', type=float, default=.01)
        self.parser.add_argument('--log_rate', type=int, default=50)
        self.parser.add_argument('--way', type=str, default='up')
        self.parser.add_argument('--ModelLogRate', type=int, default=100)

        self.parser.add_argument('--name', type=str, default='')
        self.parser.add_argument('--model_dim', type=int, default=4)

        self.parser.add_argument('--ckptPath', type=str, default='')
        self.parser.add_argument('--mixRatio', type=float, default=0)
        self.parser.add_argument('--use_moe', type=str2bool, default=False)
        self.parser.add_argument('--out_class', type=int, default=1)

        self.args = self.parser.parse_args()

    def __call__(self):
        # self.args.name = gen_name(self.args)
        if self.args.involve_loss:
            self.args.input_nodes = self.args.n_sensor + 1
        else:
            self.args.input_nodes = self.args.model_dim
        return self.args

