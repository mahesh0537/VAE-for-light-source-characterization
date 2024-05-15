import tensorflow as tf
from model import *
import copy
from load_data import *
from params import param, update_args
from utils import gen_name
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpus = tf.config.list_physical_devices('GPU')
print(f'gpus: {gpus}')
if gpus:
    for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu,True)

acc = tf.keras.metrics.Accuracy()
args_model = param()

args_model = args_model()
args_model.nbar = 2.0
args_model.max_photons = 20
args_model.n_sensor = 4
args_model.quantum_efficiency = 0.1
args_model.involve_loss = False

model_path = './checkpoints_multi/hybrid200fineTuned20240402-050158'
best_model = 1600
best_acc = 0.0

def test_model(model):
    testDataPath = '../test_data_nbarObs_1.3'
    coherent_path = os.path.join(testDataPath, 'coherent.txt')
    thermal_path = os.path.join(testDataPath, 'thermal.txt')
    listDir = []
    for i in os.walk(testDataPath):
        if len(i[1]) == 0:
            listDir.append(i[0])

    folder_mixRatio = dict()
    for path in listDir:
        mixRatio = float(path.split('_')[-1])
        if mixRatio == 1.0:
            continue
        folder_mixRatio[mixRatio] = path

    out = np.zeros((len(folder_mixRatio), len(folder_mixRatio)))
    cmats = dict()
    for mixRatio_C, path_C in folder_mixRatio.items():
        idx_C = int(mixRatio_C * 10)
        for mixRatio_T, path_T in folder_mixRatio.items():
            idx_T = int(mixRatio_T * 10)
            data_loader = load_data(args_model)
            filenames = os.listdir(path_C)
            for filename in filenames:
                if 'SPAC' in filename:
                    data_loader.appendData(os.path.join(path_C, filename))
            filenames = os.listdir(path_T)
            for filename in filenames:
                if 'SPAT' in filename:
                    data_loader.appendData(os.path.join(path_T, filename))
            data_loader.appendData(coherent_path)
            data_loader.appendData(thermal_path)
            X_test, y_test = data_loader()
            _, y_pred = model(X_test)
            y_pred_np = tf.math.argmax(y_pred, axis = 1)
            acc.reset_states()
            acc.update_state(y_test, y_pred_np)
            c_mat = confusion_matrix(y_test, y_pred_np, labels=[0,1,2,3])
            cmats[(mixRatio_C, mixRatio_T)] = c_mat
            out[idx_C, idx_T] = acc.result().numpy()
    return out

# model = model_moe(args_model.input_nodes + 1, num_experts=4)
model = model_gen(args_model.input_nodes + 1, out_class=4)
# for i in range(100, 3100, 100):
#     model.load_weights(f'{model_path}/{i}')
#     acc_mat = test_model(model)
#     # print(f'[INFO] acc {i}: {acc_mat}')
#     if acc_mat.mean() > best_acc:
#         best_acc = acc_mat.mean()
#         best_model = i
model.load_weights(f'{model_path}/{best_model}')
print(f'[INFO] best model: {best_model}')
acc_mat = test_model(model)
print(acc_mat)
np.save('acc_mat.npy', acc_mat)
print(f'[INFO] best acc: {acc_mat.mean()}')
print(f'[INFO] acc[0,5]: {acc_mat[0,5]}')


sns.heatmap(acc_mat, annot=True)
plt.xlabel('mixRatio SPAT')
plt.ylabel('mixRatio SPAC')
plt.title('Accuracy matrix')
plt.xticks(ticks = range(10), labels = [f'{i/10:.1f}' for i in range(10)])
plt.yticks(ticks = range(10), labels = [f'{i/10:.1f}' for i in range(10)])
plt.savefig('out/fig7.png', dpi=500)
plt.show()