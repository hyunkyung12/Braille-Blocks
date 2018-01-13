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
	c_img = b_img * img
	result_image = Image.fromarray(c_img)
	return result_image, b_img.reshape((img_w, img_h))

def edge_sliding_window(img, classifier, window_size, stride, boundary):
	def image_preprocessing(img):
	    from skimage import feature
	    img = np.sum(img, axis=2, keepdims=False)
	    return feature.canny(img)

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
				croped_img = np.array(croped_img).reshape((64,64,3))
				croped_img = image_preprocessing(croped_img)
				croped_img = np.array(croped_img).reshape((1,64,64,1))
				if classifier(croped_img):
					score_board = plus1(score_board, (x,y,w,h))

	b_img = (score_board / count_board) > boundary
	b_img = b_img.reshape((img_w, img_h, 1))
	c_img = b_img * img
	result_image = Image.fromarray(c_img)
	return result_image, b_img.reshape((img_w, img_h))

def score_sliding_window(img, classifier, window_size, stride, boundary, target_index, limit):
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
				scores = classifier(croped_img)
				if argmax(scores) in target_index and max(scores) >= limit:
					score_board = plus1(score_board, (x,y,w,h))

	b_img = (score_board / count_board) > boundary
	b_img = b_img.reshape((img_w, img_h, 1))
	c_img = b_img * img
	result_image = Image.fromarray(c_img)
	return result_image, b_img.reshape((img_w, img_h))

def box_search_dfs(matrix,sy,sx):
	a = matrix[sy][sx]
	i = 0
	switch = True
	while switch:
		i = i+1
		for j in range(i):
			if any([(matrix.shape[0] <= sy+i), (matrix.shape[1] <= sx+i)]):
				switch = False
				break
			if (matrix[sy+i][sx+j] != a) or (matrix[sy+j][sx+i] != a):
				switch = False
				break
	return i, i

def make_box(matrix, limit_size = (64,64)):
	box_set = []
	h, w = matrix.shape
	for y in range(h):
		for x in range(w):
			if matrix[y][x]:
				i0, i1 = box_search_dfs(matrix, y, x)
				if i0 >= limit_size[0] and i1 >= limit_size[1]:
					box_set.append([x, y, i1, i0])
					for iy in range(i0):
						for ix in range(i1):
							matrix[y+iy][x+ix] = 0
	return box_set

def hog_sliding_window(img, classifier, window_size, stride, boundary):
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
				hog = utils.hog(np.array(croped_img))
				hog = hog.reshape((-1,8,4,4))
				hog = np.transpose(hog, (0,2,3,1))
				if classifier(hog):
					score_board = plus1(score_board, (x,y,w,h))

	b_img = (score_board / count_board) > boundary
	b_img = b_img.reshape((img_w, img_h, 1))
	b_img = b_img * img
	result_image = Image.fromarray(b_img)
	return result_image


def gray_sliding_window(img, classifier, window_size, stride, boundary):
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
				croped_img = utils.grayscale(croped_img)
#				print(classifier(croped_img))
				if classifier(croped_img):
					score_board = plus1(score_board, (x,y,w,h))

	b_img = (score_board / count_board) > boundary
	b_img = b_img.reshape((img_w, img_h, 1))
	b_img = b_img * img
	result_image = Image.fromarray(b_img)
	return result_image
