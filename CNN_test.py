import tensorflow as tf
import numpy as np
import pickle
import solver
import model
import dataSet as ds

epochs = 2000
batch_size = 128
learning_rate = 1e-4
#data = pickle.load(open('bnb.p', 'rb'))

d = ds.dataSet()
#d.temp_load_data()
d.load_data(dir='image', test_dir='test_image')
d.grayscale()

sess = tf.Session()
CNN_model = model.two_layer_CNN(sess = sess, input_shape=d.train_image.shape, n_class = d.n_class)
adam_opt = solver.Solver(sess = sess, name ='Adam', dataset=d, model = CNN_model, optimizer = tf.train.AdamOptimizer)

sess.run(tf.global_variables_initializer())
adam_opt.train(epoch=epochs, batch_size=batch_size, lr=learning_rate)
adam_opt.print_result()