from model import *
from tensorflow.keras.optimizers import Adam
import datetime
from load_data import load_data
import tensorboard
from params import param, gen_data_paths   
import numpy as np
import os
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu,True)
initializer = tf.keras.initializers.GlorotNormal()

classification_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.01)

def init_model(model, initializer):
    weights = model.get_weights()
    new_weights = [initializer(w.shape) for w in weights]
    model.set_weights(new_weights)
    return model
class summary:
    def __init__(self, name = None, train_summary_writer = None):
        self.train_summary_writer = train_summary_writer
        self.mse_losses_step = tf.keras.metrics.Mean('mse loss', dtype=tf.float32)
        self.kl_losses_step = tf.keras.metrics.Mean('KL loss', dtype=tf.float32)
        self.class_loss_step = tf.keras.metrics.Mean('classification loss', dtype=tf.float32)
        self.class_accuracy = tf.keras.metrics.Mean('classification accuracy', dtype=tf.float32)
        self.test_acc = tf.keras.metrics.Mean('test accuracy', dtype=tf.float32)
        self.epoch = 0
        self.optimizer = None
        self.name = name

    def update_state(self, mse_loss, kl_loss, class_loss, class_accuracy):
        self.mse_losses_step.update_state(mse_loss)
        self.kl_losses_step.update_state(kl_loss)
        self.class_loss_step.update_state(class_loss)
        self.class_accuracy.update_state(class_accuracy)

    def reset_states(self):
        self.mse_losses_step.reset_states()
        self.kl_losses_step.reset_states()
        self.class_loss_step.reset_states()
        self.class_accuracy.reset_states()

    def update_test(self, test):
        self.test_acc.update_state(test)
    
    def __str__(self):
        return f"Epoch {self.epoch} mse loss: {self.mse_losses_step.result():.4f} KL loss: {self.kl_losses_step.result():.4f} classification loss: {self.class_loss_step.result():.4f} classification accuracy: {self.class_accuracy.result():.4f}"

    def write(self):
        if self.name is None:
            with self.train_summary_writer.as_default():
                tf.summary.scalar('mse loss', self.mse_losses_step.result(), step=self.optimizer.iterations)
                tf.summary.scalar('KL loss', self.kl_losses_step.result(), step=self.optimizer.iterations)
                tf.summary.scalar('classification loss', self.class_loss_step.result(), step=self.optimizer.iterations)
                tf.summary.scalar('classification accuracy', self.class_accuracy.result(), step=self.optimizer.iterations)
                tf.summary.scalar('test accuracy', self.test_acc.result(), step=self.optimizer.iterations)
        else:
            with self.train_summary_writer.as_default():
                tf.summary.scalar('mse loss/'+ str(self.name), self.mse_losses_step.result(), step=self.optimizer.iterations)
                tf.summary.scalar('KL loss/'+ str(self.name), self.kl_losses_step.result(), step=self.optimizer.iterations)
                tf.summary.scalar('classification loss/ '+ str(self.name), self.class_loss_step.result(), step=self.optimizer.iterations)
                tf.summary.scalar('classification accuracy/'+ str(self.name), self.class_accuracy.result(), step=self.optimizer.iterations)
                tf.summary.scalar('test accuracy/'+ str(self.name), self.test_acc.result(), step=self.optimizer.iterations)
            



def train_model(args, train_dataset, model, Xtest, ytest, train_summary_writer = None):
    optimizer = Adam(learning_rate=args.lr)
    epochs = args.epochs
    step_summary = summary(name=args.name, train_summary_writer = train_summary_writer)
    if args.out_class == 1:
        print(f'[INFO] using BinaryAccuracy')
        acc = tf.keras.metrics.BinaryAccuracy()
    else:
        print(f'[INFO] using Accuracy Matrix')
        acc = tf.keras.metrics.Accuracy()
    step_summary.optimizer = optimizer

    @tf.function
    def train_step_multi(training_batch):
        with tf.GradientTape() as tape:
            y_pred = model(training_batch[0], training=True)
            y_true = [training_batch[0], training_batch[1]]
            mse_loss = reconstruction_loss(y_true[0], y_pred[0])
            kl = sum(model.losses)
            train_loss = 0.01 * kl + mse_loss
            y_true_1 = tf.cast(y_true[1], tf.int32)
            y_true_1 = tf.one_hot(y_true_1, int(args.out_class))
            classification_loss = classification_loss_fn(y_true_1, y_pred[1])
            grads = tape.gradient([train_loss, classification_loss], model.trainable_variables)
            y_pred_np = y_pred[1]
            y_pred_np = tf.math.argmax(y_pred_np, axis = 1)
            acc.update_state(y_pred_np, training_batch[1])
            step_summary.update_state(mse_loss, kl, classification_loss, acc.result())
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return
        
    @tf.function
    def train_step_one(training_batch):
        with tf.GradientTape() as tape:
            y_pred = model(training_batch[0], training=True)
            y_true = [training_batch[0], training_batch[1]]
            mse_loss = reconstruction_loss(y_true[0], y_pred[0])
            kl = sum(model.losses)
            train_loss = 0.01 * kl + mse_loss
            classification_loss = classification_loss_fn(y_true[1], y_pred[1])
            grads = tape.gradient([train_loss, classification_loss], model.trainable_variables)
            y_pred_np = y_pred[1]
            y_pred_np = tf.where(y_pred_np >= 0.5, 1, 0)
            acc.update_state(y_pred_np, training_batch[1])
            step_summary.update_state(mse_loss, kl, classification_loss, acc.result())
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return

    if args.out_class == 1:
        print(f'[INFO] using train_step_one')
        train_step = train_step_one
    else:
        train_step = train_step_multi

    for epoch in range(1, epochs + 1):
        step_summary.reset_states()
        step_summary.epoch = epoch
        acc.reset_state()
        for step, training_batch in enumerate(train_dataset):
            train_step(training_batch)
        if epoch % args.log_rate == 0:
            print(step_summary)
        if epoch % args.ModelLogRate == 0:
            model.save_weights(os.path.join(args.ckptPath, str(epoch)))
            print('Model saved at'+ os.path.join(args.ckptPath, str(epoch)))
        step_summary.write()
    return

def main(args):
    path = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+args.name
    train_summary_writer = tf.summary.create_file_writer(path)
    data_loader = load_data(args)
    if args.use_moe:
        model = model_moe(args.input_nodes + 1, num_experts=4)
        print(f'[INFO] using MoE')
    else:
        model = model_gen(args.input_nodes + 1, out_class=args.out_class)

    #initilizing the model with he_normal
    model = init_model(model, initializer)
    try:
        model.summary()
    except Exception as e:
        pass
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    ckptPath = os.path.join(args.out_dir, args.name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))             #"SavedModel/hybrid200State"
    print(f'[INFO] ckptPath: {ckptPath}')
    if not os.path.exists(ckptPath):
        os.makedirs(ckptPath)
    args.ckptPath = ckptPath
    trainigDataPath = args.data_path
    if args.out_class == 1:
        filenames = gen_data_paths()
    else:
        filenames = os.listdir(trainigDataPath)
    for filename in filenames:
        try:
            data_loader.appendData(os.path.join(trainigDataPath, filename))
        except Exception as e:
            print('Error: ', e)
        # print('appended data from: ', filename)
        # print(data_loader)
    trainDataset, [Xtest, ytest] = data_loader()
    print(data_loader)
    train_model(args, trainDataset, model, Xtest, ytest, train_summary_writer)
    model.save_weights(ckptPath)
    print("Model Saved at: ", ckptPath)    


if __name__ == '__main__':
    args = param()()
    args.train = True
    main(args)
