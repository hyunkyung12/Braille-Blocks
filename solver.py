import tensorflow as tf
import matplotlib.pyplot as plt
import os

class Solver:
    '''
    data를 이 안에 넣을까?
    '''
    def __init__(self, sess, name, model, dataset, optimizer=tf.train.AdamOptimizer):
        self._sess = sess
        self._model = model
        self._lr = tf.placeholder(dtype=tf.float32)
        self._loss_history = []
        self._train_acc_history = []
        self._test_acc_history = []

        with tf.variable_scope(name):
            self._optimizer = optimizer(self._lr)
            self._training_op = self._optimizer.minimize(self._model._loss)

        self.dataset = dataset
        self.batch_x, self.batch_y = [], []
        self.val_x, self.val_y = self.dataset.test_image, self.dataset.test_label
        

    def train(self, epoch=200, batch_size=128, lr = 1e-2, verbose=True, print_frequency=10):
        self.batch_size = batch_size
        n_batch = self.dataset.train_size // self.batch_size
        for i, iter in enumerate(range(epoch)):
            for j in range(n_batch):
                self.batch_x, self.batch_y = self.dataset.next_batch()
                feed_train = {self._model._x: self.batch_x, self._model._y: self.batch_y, self._lr: lr}
                _, recent_loss = self._sess.run(fetches=[self._training_op, self._model._loss], feed_dict=feed_train)
                self._loss_history.append(recent_loss)
                self._train_acc_history.append(self.accuracy(self.batch_x, self.batch_y))
                self._test_acc_history.append(self.accuracy(self.val_x, self.val_y))
            if verbose:
                if i % print_frequency == print_frequency-1:
                    self._print_train_process(epoch=i+1)


    def loss(self):
        feed_loss = {self._model._x: self.batch_x, self._model._y: self.batch_y}
        return self._sess.run(fetches=self._model._loss, feed_dict=feed_loss)

    def predict(self, x_data):
        feed_predict = {self._model._x: x_data}
        return self._sess.run(fetches=self._model._prediction, feed_dict=feed_predict)

    def predict_softmax_score(self, x_data):
        feed_predict = {self._model._x: x_data}
        return self._sess.run(fetches=self._model._hypothesis, feed_dict=feed_predict)

    def print_accuracy(self, x_data, y_data):
        result = y_data == self.predict(x_data=x_data)
        print('accuracy : {:.4f}'.format(sum(result) / len(result)))

    def _print_train_process(self, epoch):
        print('epoch : {:>4}, loss : {:.4f}, train_accuracy : {:.4f}, test_accuracy : {:.4f}'.format(
            epoch, self.loss(), self.accuracy(self.batch_x, self.batch_y), self.accuracy(self.val_x, self.val_y)))

    def accuracy(self, x_data, y_data):
        if x_data is None:
            return 0
        result = y_data == self.predict(x_data)
        return sum(result) / len(result)

    def print_result(self):
        plt.plot(self._loss_history)
        plt.title('loss')
        plt.show()

        l = range(len(self._train_acc_history))
        plt.plot(l, self._train_acc_history, 'b', label = 'train_acc')
        plt.plot(l, self._test_acc_history, 'r', label = 'test_acc')
        plt.legend()
        plt.title('accuracy')
        plt.show()


    def model_save(self, save_dir="saved"):
        saver = tf.train.Saver()
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        saver.save(self._sess, save_dir+"/train")

    def model_load(self, load_dir="saved"):
        saver = tf.train.Saver()
        saver.restore(self._sess, tf.train.latest_checkpoint(load_dir))
