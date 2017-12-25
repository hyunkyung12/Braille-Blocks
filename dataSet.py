import pickle
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import json
import augmentation

class dataSet():
	'''
	 - Data 원본을 저장하고, 매번 필요한 경우 리사이징을 하자?
	'''
	def __init__(self, para_filename='para.json'):
		self.train_image = np.empty(0)
		self.train_label = np.empty(0)
		self.test_image = np.empty(0)
		self.test_label = np.empty(0)
		self.label_dic = {}
		self.load_para(para_filename)
		self.ori_img_set = []
		self.img_set = []
		self.label_set = []
		self.width, self.height, self.color = 0, 0, 0

		self.batch_size = 32
		self.batch_index = []
		self.n_example = 0
		self.n_class = 0

	def load_para(self, filename='para.json'):
		with open(filename, 'r') as f:
			js = json.load(f)
			self.label_dic = js['label_dic']
			self.resizing = js['resizing']
			self.train_size = js['train_size']
			self.test_size = js['test_size']
			self.target_list = js['target_list']
			self.score_bottom_line = js['score_bottom_line']
		self.width, self.height = self.resizing

	def load_data(self, dir=None, test_dir=None): # 현경
		sys_file = []
		self.img_set = np.empty((0,) + tuple(self.resizing) + (3,))

		# 만약 label_dic이 비어있으면, 폴더 이름을 불러옴
		if self.label_dic == {}:
			for i, filename in enumerate(os.listdir(os.getcwd() + '/' + dir)):
				self.label_dic[filename] = i

		if dir is not None:
			for label in self.label_dic:
				img_dir = os.getcwd()+ '/' + dir + '/' + label

				for path, _, files in os.walk(img_dir):
					for file in files:
						img_dir = path + '/' + file
						try:
							img = Image.open(img_dir)
						except OSError as e:
							sys_file.append(e)
						else:
							# 만약 image가 RGB 포맷이 아닐경우, RGB로 변경
							if not img.format == "RGB":
								img = img.convert("RGB")
							self.ori_img_set.append(img)
							self.img_set = np.append(self.img_set, np.array([np.array(img.resize(self.resizing))]), axis=0)
							self.label_set = np.append(self.label_set, self.label_dic[label])

			self.n_example = len(self.ori_img_set)
			self.n_class = len(self.label_dic)
			self.sep_train_test()

		else:
			self.img_set = np.empty((0,) + tuple(self.resizing) + (3,))

		if test_dir is not None:
			self.test_img = []
			img_dir = os.getcwd() + '/' + test_dir

			for path, _, files in os.walk(img_dir):
				for file in files:
					img_dir = path + '/' + file
					try:
						img = Image.open(img_dir)
					except OSError as e:
						sys_file.append(e)
					else:
						# 만약 image가 RGB 포맷이 아닐경우, RGB로 변경
						if not img.format == "RGB":
							img = img.convert("RGB")
						self.test_img.append(img)

	def sep_train_test(self):
		'''
			train / test로 분할 함, size는 para.json에 저장된 값 부름
		:return:
		'''
		ind = np.random.randint(self.n_example, size=self.train_size+self.test_size)
		self.train_image = self.img_set[ind[:self.train_size]]
		self.train_label = self.label_set[ind[:self.train_size]]
		self.test_image = self.img_set[ind[self.train_size:]]
		self.test_label = self.label_set[ind[self.train_size:]]

	def _resize(self, size=None):
		# size를 따로 받지 않았을 경우, para.json에 저장되어 있는 값으로 resizing
		if size is None:
			size = self.resizing
		resizing_size = tuple(size) + (self.img_set[0].shape[2],)
		self.img_set = np.empty((0,) + resizing_size)
		for img in self.ori_img_set:
			temp = np.array(img.resize(size)).reshape(((-1,) + resizing_size))
			self.img_set = np.append(self.img_set, temp, axis=0)

	def save(self, dir): # 현경
		temp_dataset = {
			'train_image' : self.train_image,
			'train_label' : self.train_label,
			'test_image' : self.test_image,
			'test_label' : self.test_label
		}
		with(open(dir, 'wb')) as f:
			pickle.dump(temp_dataset, f)

	def one_hot_encoding(self):  # 현경
		temp_list = np.zeros((self.n_example, self.n_class))
		temp_list[np.arange(self.n_example), np.array(self.label_set)] = 1
		self.label_set = temp_list

	def one_hot_decoding(self):
		self.label_set = [np.where(i==1)[0][0] for i in self.label_set]

	def Augmentation(self, color_aug):
		pass

	def print_informaton(self): # 광록
		print('train_data : {}, test_data : {}'.format(self.train_size, self.test_size))
		print('image_size : ({}, {})'.format(self.width, self.height))
		for i in np.unique(self.train_label):
			print('sample image : {}'.format(i))
			ind = np.where(self.train_label == i)
			self.sample_image(ind[0][0])

	def mini_batch(self, batch_size): # 광록
		pass

	def _make_batch_index(self):
		self.batch_index = np.split(np.random.permutation(self.train_size), np.arange(1, int(self.train_size // self.batch_size) + 1) * self.batch_size)
		if self.batch_index[-1] == []:
			self.batch_index = self.batch_index[:-1]

	def next_batch(self): # 광록
		if len(self.batch_index) == 0:
			self._make_batch_index()
		ind = self.batch_index[0]
		self.batch_index = self.batch_index[1:]
		return self.train_image[ind], self.train_label[ind]

	def grayscale(self):
		RGB_to_L = np.array([[[[0.299,0.587,0.114]]]])
		self.train_image = np.sum(self.train_image * RGB_to_L, axis=3, keepdims=True)
		self.test_image = np.sum(self.test_image * RGB_to_L, axis=3, keepdims=True)
		print(self.train_image.shape)

	def sample_image(self, index=0):
		if self.train_image.shape[3] == 3:
			plt.imshow(self.train_image[index]/255)
		elif self.train_image.shape[3] == 1:
			plt.imshow(self.train_image[index].reshape((self.width, self.height))/255)
		plt.show()

	def object_detect(self):
		pass