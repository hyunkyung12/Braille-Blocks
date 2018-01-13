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
import augmentation
from xml.etree.ElementTree import ElementTree

'''
selective search에 들어갈 이미지는 np객체
'''

parser = argparse.ArgumentParser()
parser.add_argument("-training", "--training", type=bool, default=False)
parser.add_argument("-epochs", "--epochs", type=int, default=2000)
args = parser.parse_args()

print("Phase 0 : Load data")
data = ds.dataSet()
j = ds.jsonObject('para.json')
param = j.para

target_index = []
for cl in param['label_dic']:
    if cl in param['target_list']:
        target_index.append(param['label_dic'][cl])

training = args.training

if training:
    data_dir = 'image'
else:
    data_dir = None
data.load_data(dir=data_dir, test_dir='test_image')

#data.grayscale()
data.augmentation()
#data.edge()

if training:
    data.sep_train_test()
_, *model_input_size = data.img_set.shape

sess = tf.Session()
model = model.four_layer_CNN(sess=sess, input_shape=data.img_set.shape, n_class=len(param['label_dic']))
sv = solver.Solver(sess=sess, name='op', model=model, dataset=data, optimizer=tf.train.AdamOptimizer)

epochs = args.epochs
batch_size = param['batch_size']
learning_rate = param['lr']
path = '/home/paperspace/Dropbox/krlee/easy-yolo/devkit/2017/Images'

sess.run(tf.global_variables_initializer())

if not training:
    print("Phase 1 : Load model")
    sv.model_load()

else:
    print("Phase 1 : Train model")
    sv.train(epoch=epochs, batch_size=batch_size, lr=learning_rate, print_frequency=100)
    sv.model_save()

def cf(x):
    return sv.predict(x)[0] in target_index

def score_cf(imag):
    scores = np.array(sv.predict_softmax_score(imag))
    return (np.argmax(scores) in target_index) and (np.max(scores) > param['score_bottom_line'])

for i, img in enumerate(data.test_img):
    print("{}th image in progress".format(i))
    temp_img, matrix = sw.sliding_window(img, score_cf, window_size=param['ss_dic']['window_size'], stride=16, boundary=param['sw_dic']['boundary'])
    temp_img.save('./test_result/sw'+str(i)+'.jpg', 'jpeg')
    img.save('./Images/sw'+str(i)+'.jpg', 'jpeg')
    box_set = sw.make_box(matrix)
    for box in box_set:
        img = utils.draw_rectangle(img, box, label='block')
    xml = utils.box_to_xml('sw'+str(i)+'.xml', path+str(i)+'.jpg', img.size+(3,), box_set)
    ElementTree(xml).write('./Annotation/sw'+str(i)+'.xml')
    txt = utils.box_to_txt([0 for _ in range(len(box_set))], box_set, img.size)
    with open('./labels/sw'+str(i)+'.txt', 'w') as f:
        f.write(txt)
    img.save('./test_result/sw_box'+str(i)+'.jpg', 'jpeg')