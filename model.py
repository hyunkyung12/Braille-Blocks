import tensorflow as tf

class two_layer_CNN:
    '''
     - X : 4차원 이미지 데이터, 사이즈에 유동적
     - y : 1차원 class 데이터
    '''
    def __init__(self, sess, input_shape, n_class,
                 activation_fn=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer()):

        _, w, h, d = input_shape
        self._sess = sess
        self._x = tf.placeholder(tf.float32, [None, w, h, d])
        self._y = tf.placeholder(tf.int32, [None])
        y_one_hot = tf.one_hot(self._y, n_class)
        y_one_hot = tf.reshape(y_one_hot, [-1, n_class])

        W1 = tf.get_variable(name="W1", shape=[3, 3, d, 32], dtype=tf.float32, initializer=initializer)
        L1 = tf.nn.conv2d(self._x, W1, strides=[1, 1, 1, 1], padding='SAME')
        L1 = activation_fn(L1)
        L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        w = int(w / 2 + 0.5)
        h = int(h / 2 + 0.5)

        W2 = tf.get_variable(name="W2", shape=[3, 3, 32, 64], dtype=tf.float32, initializer=initializer)
        L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
        L2 = activation_fn(L2)
        L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        w = int(w / 2 + 0.5)
        h = int(h / 2 + 0.5)

        L2_flat = tf.reshape(L2, [-1, w * h * 64])

        W3 = tf.get_variable("W3", shape=[w * h * 64, n_class], initializer=initializer)
        b = tf.Variable(tf.random_normal([n_class]))
        logits = tf.matmul(L2_flat, W3) + b
        self._hypothesis = tf.nn.softmax(logits)
        self._prediction = tf.argmax(input=logits, axis=-1)

        self._xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_one_hot)
        self._loss = tf.reduce_mean(self._xentropy, name="loss")

class three_layer_CNN:
    '''
     - X : 4차원 이미지 데이터, 사이즈에 유동적
     - y : 1차원 class 데이터
    '''
    def __init__(self, sess, input_shape, n_class,
                 activation_fn=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer()):

        _, w, h, d = input_shape
        self._sess = sess
        self._x = tf.placeholder(tf.float32, [None, w, h, d])
        self._y = tf.placeholder(tf.int32, [None])
        y_one_hot = tf.one_hot(self._y, n_class)
        y_one_hot = tf.reshape(y_one_hot, [-1, n_class])

        W1 = tf.get_variable(name="W1", shape=[3, 3, d, 32], dtype=tf.float32, initializer=initializer)
        L1 = tf.nn.conv2d(self._x, W1, strides=[1, 1, 1, 1], padding='SAME')
        L1 = activation_fn(L1)
        L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        w = int(w / 2 + 0.5)
        h = int(h / 2 + 0.5)

        W2 = tf.get_variable(name="W2", shape=[3, 3, 32, 64], dtype=tf.float32, initializer=initializer)
        L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
        L2 = activation_fn(L2)
        L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        w = int(w / 2 + 0.5)
        h = int(h / 2 + 0.5)

        W3 = tf.get_variable(name="W3", shape=[3, 3, 64, 32], dtype=tf.float32, initializer=initializer)
        L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
        L3 = activation_fn(L3)
        L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        w = int(w / 2 + 0.5)
        h = int(h / 2 + 0.5)

        L3_flat = tf.reshape(L3, [-1, w * h * 32])

        W4 = tf.get_variable("W4", shape=[w * h * 32, n_class], initializer=initializer)
        b = tf.Variable(tf.random_normal([n_class]))
        logits = tf.matmul(L3_flat, W4) + b
        self._hypothesis = tf.nn.softmax(logits)
        self._prediction = tf.argmax(input=logits, axis=-1)

        self._xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_one_hot)
        self._loss = tf.reduce_mean(self._xentropy, name="loss")


class four_layer_CNN:
    '''
     - X : 4차원 이미지 데이터, 사이즈에 유동적
     - y : 1차원 class 데이터
    '''
    def __init__(self, sess, input_shape, n_class,
                 activation_fn=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer()):

        _, w, h, d = input_shape
        self._sess = sess
        self._x = tf.placeholder(tf.float32, [None, w, h, d])
        self._y = tf.placeholder(tf.int32, [None])
        y_one_hot = tf.one_hot(self._y, n_class)
        y_one_hot = tf.reshape(y_one_hot, [-1, n_class])

        W1 = tf.get_variable(name="W1", shape=[3, 3, d, 32], dtype=tf.float32, initializer=initializer)
        L1 = tf.nn.conv2d(self._x, W1, strides=[1, 1, 1, 1], padding='SAME')
        L1 = activation_fn(L1)
        L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        w = int(w / 2 + 0.5)
        h = int(h / 2 + 0.5)

        W2 = tf.get_variable(name="W2", shape=[3, 3, 32, 64], dtype=tf.float32, initializer=initializer)
        L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
        L2 = activation_fn(L2)
        L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        w = int(w / 2 + 0.5)
        h = int(h / 2 + 0.5)

        W3 = tf.get_variable(name="W3", shape=[3, 3, 64, 64], dtype=tf.float32, initializer=initializer)
        L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
        L3 = activation_fn(L3)
        L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        w = int(w / 2 + 0.5)
        h = int(h / 2 + 0.5)

        W4 = tf.get_variable(name="W4", shape=[3, 3, 64, 32], dtype=tf.float32, initializer=initializer)
        L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
        L4 = activation_fn(L4)
        L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        w = int(w / 2 + 0.5)
        h = int(h / 2 + 0.5)

        L4_flat = tf.reshape(L4, [-1, w * h * 32])

        W5 = tf.get_variable("W5", shape=[w * h * 32, n_class], initializer=initializer)
        b = tf.Variable(tf.random_normal([n_class]))
        logits = tf.matmul(L4_flat, W5) + b
        self._hypothesis = tf.nn.softmax(logits)
        self._prediction = tf.argmax(input=logits, axis=-1)

        self._xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_one_hot)
        self._loss = tf.reduce_mean(self._xentropy, name="loss")