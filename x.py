
# Neuro Network and Deep learning with Stochastic Gradient Descent.
# Problems:
# 1. slow learning problem: cost fn with cross entropy, random init weight.
# 2. overfitting: regulation with weigt decay, Relu, learn rate, 
# 3. model parameters : random init weight.
# 4. hyper parameters : batch size, learning rate, epsilong, etc.
#
>>> import mnist_loader
>>> training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
>>> import network2
>>> net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
>>> net.large_weight_initializer()
>>> net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)

>>> import mnist_loader 
>>> training_data, validation_data, test_data = mnist_loader.load_data_wrapper() 
>>> import network2 
>>> net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
>>> net.large_weight_initializer()
>>> net.SGD(training_data[:1000], 400, 10, 0.5,
    evaluation_data=test_data, lmbda = 0.1,
    monitor_evaluation_cost=True, monitor_evaluation_accuracy=True,
    monitor_training_cost=True, monitor_training_accuracy=True)


>>> import network3
>>> from network3 import Network
>>> from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
>>> training_data, validation_data, test_data = network3.load_data_shared()
>>> mini_batch_size = 10
>>> net = Network([
    FullyConnectedLayer(n_in=784, n_out=100),
    SoftmaxLayer(n_in=100, n_out=10)], 
    mini_batch_size)
>>> net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)

>>> net = Network([
  ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                filter_shape=(20, 1, 5, 5), 
                poolsize=(2, 2)),
  ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), 
                filter_shape=(40, 20, 5, 5), 
                poolsize=(2, 2)),
  FullyConnectedLayer(n_in=40*4*4, n_out=100),
  SoftmaxLayer(n_in=100, n_out=10)], 
  mini_batch_size)
>>> net.SGD(training_data, 60, mini_batch_size, 0.1, 
            validation_data, test_data)


>>> from network3 import ReLU
>>> net = Network([
    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                  filter_shape=(20, 1, 5, 5), 
                  poolsize=(2, 2), 
                  activation_fn=ReLU),
    ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), 
                  filter_shape=(40, 20, 5, 5), 
                  poolsize=(2, 2), 
                  activation_fn=ReLU),
    FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
    SoftmaxLayer(n_in=100, n_out=10)],
    mini_batch_size)
>>> net.SGD(training_data, 60, mini_batch_size, 0.03, 
            validation_data, test_data, lmbda=0.1)

>>> net = Network([
    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                  filter_shape=(20, 1, 5, 5), 
                  poolsize=(2, 2), 
                  activation_fn=ReLU),
    ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), 
                  filter_shape=(40, 20, 5, 5), 
                  poolsize=(2, 2), 
                  activation_fn=ReLU),
    FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
    FullyConnectedLayer(n_in=100, n_out=100, activation_fn=ReLU),
    SoftmaxLayer(n_in=100, n_out=10)],
    mini_batch_size)
>>> net.SGD(expanded_training_data, 60, mini_batch_size, 0.03, 
            validation_data, test_data, lmbda=0.1)

>>> net = Network([
    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                  filter_shape=(20, 1, 5, 5), 
                  poolsize=(2, 2), 
                  activation_fn=ReLU),
    ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), 
                  filter_shape=(40, 20, 5, 5), 
                  poolsize=(2, 2), 
                  activation_fn=ReLU),
    FullyConnectedLayer(
        n_in=40*4*4, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
    FullyConnectedLayer(
        n_in=1000, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
    SoftmaxLayer(n_in=1000, n_out=10, p_dropout=0.5)], 
    mini_batch_size)
>>> net.SGD(expanded_training_data, 40, mini_batch_size, 0.03, 
            validation_data, test_data)



####
# tensorflow 
####

1. reshape input match weight variable shape.
2. Take a img 4-d input and computes a 2-D convolution graph with weight and bias.
  
  The convolution ops sweep a 2-D filter over a batch of images, 
  applying the filter to each window of each image of the appropriate size. 
  The different ops trade off between generic vs. specific filters:

  [batch, in_height, in_width, in_channels] => [ht*width*chan, out-chan]
  tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
  tf.nn.conv2d(x_image, W=[shape], strides=[], )

  Given an input tensor of shape [batch, in_height, in_width, in_channels] 
  and a filter/kernel tensor of shape [filter_height, filter_width, in_channels, out_channels], 
  this op performs the following:
  
  Flattens the filter to a 2-D matrix with shape [filter_height * filter_width * in_channels, output_channels].
  Extracts image patches from the input tensor to form 
  a virtual tensor of shape 
  [batch, out_height, out_width, filter_height * filter_width * in_channels].
  For each patch, right-multiplies the filter matrix and the image patch vector.

3. apply Relu activation.
  h_conv = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
  The pooling ops sweep a rectangular window over the input tensor, computing 
  a reduction operation for each window (average, max, or max with argmax). 
  Each pooling op uses rectangular windows of size ksize separated by offset strides. 
  For example, if strides is all ones every window is used, if strides is all twos 
  every other window is used in each dimension, etc.
  h_pool = tf.nn.max_pool(h_conv)

3. connect multiple layers
4. fully connected layers Vs. spatial
  reshape prev 64 features with 7x7 block, and output 1024 feature.
    w_fc1 = weight_variable([7x7x64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool_flat = tf.reshape(h_pool2, [-1, 7x7x64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, w_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, tf.placeholder(tf.float32))

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


4. compute cross_entropy with y_ as argument
5. train_step = optimize fn to minimize cross_entropy
6. train the model with batches and feed_dict{x:, y_:}

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.initialize_all_variables())
    for i in range(20000):
      batch = mnist.train.next_batch(50)
      if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
          x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

vanilla linear regression does not have hyperparameters. Variants of line regression does.
hyperparameters control the capacity of the model. i.e., how flexible the model is.
how many degrees of freedom it has in fitting the data.

capacity of the model., how flexible the model is.how many degrees of freedom the model.
overfitting: model adapts too much to the training data, makes the model not applicable
to other data.

training a model means using an optimization procedure to determine the best model parameters
to fit the data.


import tensorflow as tf
x = tf.placeholder(tf.float32, [None, 784])
# model parameters as variables
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
# true value in _y
y_ = tf.placeholder(tf.float32, [None, 10])
# cross-entropy sum(y_*lg(y))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# each train step, stochastic Gradient Descend to optimize cross_entropy
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


""" Deep learning convolutional Network """
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# all layers use Rectifier Linear Unit
x_image = tf.reshape(x, [-1,28,28,1])
# 1st conv layer using Rectifier Linear Unit
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
# 2nd conv layer take first conv layer output as input
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# fully connected layer reshape 2nd layer output
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# apply dropout regulation to reduce overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# finally, add softmax layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
# reduce mean of network output
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
# each train step optimize minimal cross entropy
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
sess.run(tf.initialize_all_variables())

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


"""
tf.contrib.learn.[datasets, DNNClassifier,]
1. Load CSVs containing Iris training/test data into a TensorFlow Dataset
2. Construct a neural network classifier with DNNClassifier(hidden_units=[])
3. Fit the model using the training data
4. Evaluate the accuracy of the model
5. Classify new samples
"""
import tensorflow as tf
import numpy as np
# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

# train and test datasets, no validate set.
training_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TRAINING, target_dtype=np.int)
test_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TEST, target_dtype=np.int)
x_train, x_test, y_train, y_test = training_set.data, test_set.data, training_set.target, test_set.target

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=3)
# Fit model with x=train.data and y=train.result
classifier.fit(x=x_train, y=y_train, steps=200)

# Evaluate accuracy.
accuracy_score = classifier.evaluate(x=x_test, y=y_test)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

# Classify two new flower samples.
new_samples = np.array(
  [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
y = classifier.predict(new_samples)
print ('Predictions: {}'.format(str(y)))


Linear models assign independent weights to features.
  train quickly, compared to deep neural nets.
  can work well on very large feature sets.
  can be trained with algorithms that dont require a lot of fiddling with learning rates, etc.
  can be interpreted and debugged more easily than neural nets.
  provide an excellent starting point for learning about machine learning.
  are widely used in industry.

FeatureColumn repr a single feature[continuous feature, categorical feature] in your data.
Sparse columns repr sparse vector from categorical feature.
  SparseColumn
  RealValuedColumn
  BucketizedColumn
  CrossedColumn

Continuous feature:
  age = tf.contrib.layers.real_valued_column("age")
Bucketization:
  age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

Encoding sparse columns with sparse_column_with_keys.
  eye_color = tf.contrib.layers.sparse_column_with_keys(column_name="eye_color", keys=["blue", "brown", "green"])
  education = tf.contrib.layers.sparse_column_with_hash_bucket("education", hash_bucket_size=1000)
Derived feature columns with crossed_column.
  sport = tf.contrib.layers.sparse_column_with_hash_bucket("sport", hash_bucket_size=1000)
  city = tf.contrib.layers.sparse_column_with_hash_bucket("city", hash_bucket_size=1000)
  sport_x_city = tf.contrib.layers.crossed_column([sport, city], hash_bucket_size=int(1e4))
  age_buckets_x_race_x_occupation = tf.contrib.layers.crossed_column(
    [age_buckets, race, occupation], hash_bucket_size=int(1e6))

The input fn rets a map of tensors.
key is the name of FeatureColumn and value is constant tensor. {k: tf.constant(v)}

  def input_fn(df):
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values,
        shape=[df[k].size, 1])
                        for k in CATEGORICAL_COLUMNS}
    # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols.items() + categorical_cols.items())
    # Converts the label column into a constant Tensor.
    label = tf.constant(df[LABEL_COLUMN].values)
    # Returns the feature columns and the label.
    return feature_cols, label

  def train_input_fn():
    return input_fn(df_train)

  def eval_input_fn():
    return input_fn(df_test)

LinearClassifier with FeatureColumns and model, and l1 l2 regulation.
  e = tf.contrib.learn.LinearClassifier(feature_columns=[
  native_country, education, occupation, workclass, marital_status,
  race, age_buckets, education_x_occupation, age_buckets_x_race_x_occupation],
  optimizer=tf.train.FtrlOptimizer(
    learning_rate=0.1,
    l1_regularization_strength=1.0,
    l2_regularization_strength=1.0),
  model_dir=YOUR_MODEL_DIRECTORY)

  e.fit(input_fn=input_fn_train, steps=200)
  # Evaluate for one step (one pass through the test data).
  results = e.evaluate(input_fn=input_fn_test, steps=1)

Wide model is a linear model with a wide set of sparse and crossed feature columns.
  wide_columns = [
    gender, native_country, education, occupation, workclass, marital_status, relationship, age_buckets,
    tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=int(1e4)),
    tf.contrib.layers.crossed_column([native_country, occupation], hash_bucket_size=int(1e4)),
    tf.contrib.layers.crossed_column([age_buckets, race, occupation], hash_bucket_size=int(1e6))]

Deep columns are categorical columns using embedding_columns + continuous columns.
  deep_columns = [
      tf.contrib.layers.embedding_column(workclass, dimension=8),
      tf.contrib.layers.embedding_column(education, dimension=8),
      tf.contrib.layers.embedding_column(marital_status,dimension=8),
      tf.contrib.layers.embedding_column(gender, dimension=8),
      tf.contrib.layers.embedding_column(relationship, dimension=8),
      tf.contrib.layers.embedding_column(race, dimension=8),
      tf.contrib.layers.embedding_column(native_country,dimension=8),
      tf.contrib.layers.embedding_column(occupation, dimension=8),
      age,
      education_num,
      capital_gain,
      capital_loss,
      hours_per_week,
  ]

if FLAGS.model_type == "wide":
    m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                          feature_columns=wide_columns)
  elif FLAGS.model_type == "deep":
    m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                       feature_columns=deep_columns,
                                       hidden_units=[100, 50])
  else:
    '''Fully connected hidden layers to find every combination of pair of features.'''
    m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50])
  return m

m.fit(input_fn=train_input_fn, steps=200)
results = m.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print "%s: %s" % (key, results[key])


##############################
# Exporter(train.Saver) module to export trained model and graph.
# session.graph.as_graph_def() ret serialized model graph.
##############################
from tensorflow_serving.session_bundle import exporter
export_path = sys.argv[-1]
saver = tf.train.Saver(sharded=True)
model_exporter = exporter.Exporter(saver)
signature = exporter.classification_signature(input_tensor=x, scores_tensor=y)
model_exporter.init(sess.graph.as_graph_def(), default_graph_signature=signature)
model_exporter.export(export_path, tf.constant(FLAGS.export_version), sess)



while stk or cur:
  if cur:
    stk.append(cur)
    cur = cur.left
  else:
    cur = stk.pop
    visite(cur)
    cur = cur.rite

