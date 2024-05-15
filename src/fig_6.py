from load_data import load_data
from params import param
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from model import model_gen
def gen_name(args):
    return "{}_{}_{}_{}_{}".format(args.nbar, args.involve_loss, args.n_sensor, args.quantum_efficiency, args.max_photons)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpus = tf.config.list_physical_devices('GPU')
print(f'gpus: {gpus}')
if gpus:
    for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu,True)

acc = tf.keras.metrics.Accuracy()

SavedModel = model_gen(input_nodes=6)
SavedModel.load_weights('./checkpoints_one/spac_spat20240515-175006/3000')

def test(testData):
    out = SavedModel(testData[0])[1]
    out = tf.where(out > 0.5, 1, 0)
    acc.reset_state()
    acc.update_state(out, testData[1])
    return acc.result().numpy()


def genPlotData(nbar, qe):
    args = param(ipynb=True)()
    args.genTrainingData = False
    args.nbar = nbar
    args.quantum_efficiency = qe
    args.train = False
    args.name = gen_name(args)
    args.path = "../testData/nbar"+str(args.name)
    # dataLoader = load_data(args)
    bin_size = 200
    dataLoader = load_data(args)
    dataLoader.load(bin_size)
    testData = dataLoader()
    return [dataLoader.nbar, test(testData)]

outDatas = []
for qe in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    for nbar in [1.9, 2.0]:
        outDatas.append(genPlotData(nbar, qe))
outDatas = np.array(outDatas)
outDatas = outDatas[(outDatas.T)[0].argsort()]
plt.plot(outDatas[:,0], outDatas[:,1])
plt.scatter(outDatas[:, 0], outDatas[:,1])
plt.ylabel('Accuracy')
plt.xlabel('Nbar Observed')
plt.title('Nbar Observed vs Accuracy')
plt.savefig('out/fig6.png')
plt.show()
