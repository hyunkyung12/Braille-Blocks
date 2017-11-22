import dataSet as ds
import selectivesearch
import numpy as np
import utils
import model
import solver
import tensorflow as tf
from PIL import Image
'''
selective search에 들어갈 이미지는 np객체

'''

print("Phase 0 : Load data")
data = ds.dataSet()
data.load_data(dir='image', test_dir='test_image')
#data.grayscale()
_, *model_input_size = data.train_image.shape

sess = tf.Session()

model = model.two_layer_CNN(sess=sess, input_shape=data.train_image.shape, n_class=2)
sv = solver.Solver(sess=sess, name='op', model=model, dataset=data, optimizer=tf.train.AdamOptimizer)

epochs = 2000
batch_size = 128
learning_rate = 1e-4

sess.run(tf.global_variables_initializer())
print("Phase 1 : Training model")
sv.train(epoch=epochs, batch_size=batch_size, lr=learning_rate, print_frequency=100)
sv.print_result()

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
    labeled_image = np.empty(0)
    for j, c_img in enumerate(croped_images):
        c_img = c_img.resize(model_input_size[0:2])
        #c_img = c_img.convert('L')
        model_input = np.array(c_img).reshape((1,) + tuple(model_input_size))
        if sv.predict(model_input):
            labeled_image = np.append(labeled_image, j)

    labeled_image = np.int_(labeled_image)
    temp_img = img.copy()
    for ind in labeled_image:
        temp_img = utils.draw_rectangle(temp_img, regions[ind], label='Braille-Blocks')
    temp_img.save('./test_result/classfication'+str(i)+'.jpg')