import dataSet as ds
import selectivesearch
import numpy as np
import utils
import model
import solver
import tensorflow as tf
from PIL import Image
import argparse
import slidingWindow as sw

'''
selective search에 들어갈 이미지는 np객체
'''

parser = argparse.ArgumentParser()
parser.add_argument("-training", "--training", type=bool, default=False)
parser.add_argument("-epochs", "--epochs", type=int, default=2000)
args = parser.parse_args()

print("Phase 0 : Load data")
data = ds.dataSet()

training = args.training

if training:
    data_dir = 'image'
else:
    data_dir = None

data.load_data(dir=data_dir, test_dir='test_image')
#data.grayscale()
_, *model_input_size = data.img_set.shape

sess = tf.Session()

model = model.two_layer_CNN(sess=sess, input_shape=data.img_set.shape, n_class=2)
sv = solver.Solver(sess=sess, name='op', model=model, dataset=data, optimizer=tf.train.AdamOptimizer)

epochs = args.epochs
batch_size = 128
learning_rate = 1e-4

sess.run(tf.global_variables_initializer())

if not training:
    print("Phase 1 : Load model")
    sv.model_load()

else:
    print("Phase 1 : Train model")
    sv.train(epoch=epochs, batch_size=batch_size, lr=learning_rate, print_frequency=100)
    sv.print_result()
    sv.model_save()

for i, img in enumerate(data.test_img):
    print("{}th image in progress".format(i))
    temp_img = sw.sliding_window(img, sv.predict, window_size=[[64,64]], stride=16, boundary=0.3)
    temp_img.save('./test_result/sw'+str(i)+'.jpg', 'jpeg')