import tensorflow as tf
from tensorflow.keras.activations import selu
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Lambda, Multiply, Add, LeakyReLU, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential

def reconstruction_loss(y, y_pred):
    return tf.reduce_mean(tf.square(y - y_pred))

def kl_loss(mu, log_var):
    loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mu) - tf.exp(log_var))
    return loss

def vae_loss(y_true, y_pred, mu, log_var):
    return reconstruction_loss(y_true, y_pred) + (1 / (64*64)) * kl_loss(mu, log_var)

def model_gen(input_nodes = 4, if_testing = False, if_moe = False, out_class = 1):
    inputs = Input((input_nodes,))
    act_cnn = 'tanh'
    x = Dense(16, activation = selu)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation = selu)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation = selu)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation = selu)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation = selu)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    d = 8
    x = Dense(d, activation = selu)(x)
    encoder_output = BatchNormalization()(x)
    latent_dim = 3
    mu = Dense(latent_dim, name='mu')(encoder_output)
    log_var = Dense(latent_dim, name = 'log_var')(encoder_output)

    epsilon = Lambda(lambda mu: K.random_normal(shape = (tf.shape(mu)[0], tf.shape(mu)[1])))(mu)
    sigma = Lambda(lambda log_var : tf.exp(0.5 * log_var))(log_var)

    z_eps = Lambda((lambda x : Multiply()([x[0], x[1]])), name = 'z_eps')([sigma, epsilon])
    z = Lambda(lambda x : Add()([x[0], x[1]]), name= 'z')([mu, z_eps])
    encoder = Model(inputs, outputs = [mu, log_var, z], name = 'encoder')
    classifier = Sequential(name = 'Classifier')
    classifier_layers = [8, 16, 16] if out_class == 1 else [16, 64, 64]
    classifier.add(Dense(classifier_layers[0], input_shape = (latent_dim,)))
    classifier.add(LeakyReLU())
    classifier.add(Dropout(0.2))
    classifier.add(BatchNormalization())
    classifier.add(Dense(classifier_layers[1]))
    classifier.add(LeakyReLU())
    classifier.add(Dropout(0.2))
    classifier.add(BatchNormalization())
    classifier.add(Dense(classifier_layers[2]))
    classifier.add(LeakyReLU())
    classifier.add(BatchNormalization())
    classifier.add(Dropout(0.2))
    if if_moe:
        classifier.add(Dense(out_class + 1, activation = 'relu'))
    elif out_class == 4:
        classifier.add(Dense(out_class, activation = 'softmax'))
    else:
        classifier.add(Dense(out_class, activation = 'relu'))
    dr = 0.2
    decoder = Sequential(name = 'Decoder')
    decoder_layers = [16, 32, 16] if out_class == 1 else [16, 32, 64, 32, 16]
    decoder.add(Dense(8, activation = selu, input_shape = (latent_dim, )))
    decoder.add(Dropout(dr))
    decoder.add(BatchNormalization())
    for i in range(len(decoder_layers)):
        decoder.add(Dense(decoder_layers[i], activation = selu))
        decoder.add(Dropout(dr))
        decoder.add(BatchNormalization())
    decoder.add(Dense(input_nodes, activation = 'relu'))
    mu, log_var, z = encoder(inputs)
    reconstructed = decoder(z)
    classification = classifier(z)
    model = Model(inputs, [reconstructed, classification], name ="vae")
    print(classifier.summary())
    # model_testing = Model(inputs, [reconstructed, classification, mu, log_var, z], name = 'vae_testing')
    if if_testing:
        model = Model(inputs, classification, name = 'testing')

    loss = kl_loss(mu, log_var)
    model.add_loss(loss)
    return model                #, model_testing

class model_moe(tf.keras.Model):
    def __init__(self, input_nodes = 4, num_experts = 8):
        super(model_moe, self).__init__()
        self.input_nodes = input_nodes
        self.num_experts = num_experts
        self.experts = [model_gen(input_nodes, if_moe=True, out_class= 4) for i in range(num_experts)]
        self.gate = tf.keras.layers.Softmax()
        self.classification_softmax = tf.keras.layers.Softmax()

    def call(self, inputs):
        experts_output = [expert(inputs) for expert in self.experts]
        expert_classifications  = [expert[1] for expert in experts_output]
        gate = tf.stack([expert[:,0] for expert in expert_classifications], axis = 1)
        gate_output = self.gate(gate)
        expert_reconstructions = tf.reduce_sum([tf.multiply(tf.expand_dims(gate_output[:, i], axis=1), experts_output[i][0]) for i in range(self.num_experts)], axis = 0)
        expert_classifications = tf.reduce_sum([tf.multiply(tf.expand_dims(gate_output[:, i], axis=1), experts_output[i][1][:,1:]) for i in range(self.num_experts)], axis = 0)
        expert_classifications = self.classification_softmax(expert_classifications)
        return expert_reconstructions, expert_classifications