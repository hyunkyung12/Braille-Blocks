import numpy as np
from PIL import Image
import utils

def plus1(np_array, region):
	aw, ah = np_array.shape
	x, y, w, h = region
	X = np.array([[i for i in range(ah)] for _ in range(aw)])
	Y = np.array([[i for _ in range(ah)] for i in range(aw)])
	X = (X >= x) & (X < x+w)
	Y = (Y >= y) & (Y < y+h)
	XY = X & Y
	return(np_array + XY)

def sliding_window(img, classifier, window_size, stride, boundary):
	np_img = np.array(img)
	img_w, img_h, _ = np_img.shape
	score_board = np.zeros(np_img.shape[0:2])
	count_board = np.zeros(np_img.shape[0:2])
	for window in window_size:
		w, h = window
		for x in range(0, img_h-h, stride):
			for y in range(0, img_w-w, stride):
				count_board = plus1(count_board, (x,y,w,h))
				croped_img = utils.seperate_region(img, (x, y, w, h))
				croped_img = croped_img.resize((64,64))
				croped_img = np.array(croped_img).reshape((1,64,64,3))
				if classifier(croped_img):
					score_board = plus1(score_board, (x,y,w,h))

	b_img = (score_board / count_board) > boundary
	b_img = b_img.reshape((img_w, img_h, 1))
	b_img = b_img * img
	result_image = Image.fromarray(b_img)
	return result_image