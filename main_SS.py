import dataSet as ds
import selectivesearch
import numpy as np
import utils
import model
import solver
import tensorflow as tf
from PIL import Image
import argparse
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
    np_img = np.array(img)
    print("Image {}, Phase 2 : Selective search".format(i+1))
    img_lbl, ss_regions = selectivesearch.selective_search(np_img, scale=100, sigma=0.5, min_size=10)

    regions = utils.refining_ss_regions(ss_regions)
    croped_images = []
    temp_img = img.copy()
    for region in regions:
        temp_img = utils.draw_rectangle(temp_img, region)
        croped_images.append(utils.seperate_region(img, region))
    temp_img.save('./test_result/'+str(i)+'.jpg', 'jpeg')

    print("Image {}, Phase 3 : Classification".format(i+1))
    #labeled_image = np.empty(0)
    temp_img = img.copy()
    for j, c_img in enumerate(croped_images):
        c_img = c_img.resize(model_input_size[0:2])
        model_input = np.array(c_img).reshape((1,) + tuple(model_input_size))
        softmax_score = sv.predict_softmax_score(model_input)[0]
        target_index = np.argmax(softmax_score)
        if softmax_score[target_index] > data.score_bottom_line \
                and list(data.label_dic.keys())[target_index] in data.target_list:
            temp_img = utils.draw_rectangle(temp_img, regions[j], label=list(data.label_dic.keys())[target_index])

    temp_img.save('./test_result/classfication'+str(i)+'.jpg')