import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

def unpickle(file):
    # load pickled data
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict

def accuracy(pred, labels):
    return (100.0 * np.sum(np.argmax(pred, 1) == np.argmax(labels, 1))/ pred.shape[0])


if __name__=='__main__':
    train_fnroot = "cifar-10-batches-py/data_batch_"
    test_filename = "cifar-10-batches-py/test_batch"
    meta_filename = "cifar-10-batches-py/batches.meta"

    train_dateset = None
    train_labels = None
    for i in range(1,6):
        train_filename = train_fnroot + str(i)
        batch = unpickle(train_filename)
        if i==1:
            train_dataset = batch['data']
            train_labels = np.array(batch['labels'])
        else:
            train_dataset = np.concatenate((train_dataset, batch['data']), axis=0)
            train_labels = np.concatenate((train_labels, batch['labels']))
    del batch

    train_dataset, valid_dataset, train_labels, valid_labels = train_test_split(train_dataset, train_labels, test_size=10000, random_state=897, stratify=train_labels)

    # Load Test Dataset
    test_batch = unpickle(test_filename)
    test_dataset = test_batch['data']
    test_labels = np.array(test_batch['labels'])
    del test_batch
    # Load Label Names
    label_names = unpickle(meta_filename)['label_names']

    print 'Dataset\t\tFeatureShape\tLabelShape'
    print 'Training set:\t', train_dataset.shape,'\t', train_labels.shape
    print 'Validation set:\t', valid_dataset.shape,'\t', valid_labels.shape
    print 'Testing set:\t', test_dataset.shape, '\t', test_labels.shape

    # Reshape the data into pixel by pixel by RGB channels
    train_reshape = np.rollaxis(train_dataset.reshape((-1,3,32,32)), 1, 4)
    valid_reshape = np.rollaxis(valid_dataset.reshape((-1,3,32,32)), 1, 4)
    test_reshape = np.rollaxis(test_dataset.reshape((-1,3,32,32)), 1, 4)
    print 'Dataset\t\tFeatureShape\t\tLabelShape'
    print 'Training set:\t', train_reshape.shape,'\t', train_labels.shape
    print 'Validation set:\t', valid_reshape.shape, '\t', valid_labels.shape
    print 'Testing set:\t', test_reshape.shape, '\t', test_labels.shape

    # Mirror Reflection
    train_LRF = train_reshape[:,:,::-1,:]

    # Prepare Data
    image_size = 32
    num_labels = 10
    num_channels = 3

    train_dataset = np.concatenate((train_reshape, train_LRF), 0).astype(np.float32)
    train_labels = (np.arange(num_labels)==train_labels[:,None]).astype(np.float32)
    del train_reshape
    del train_LRF

    valid_dataset = valid_reshape.astype(np.float32)
    valid_labels = (np.arange(num_labels)==valid_labels[:, None]).astype(np.float32)
    del valid_reshape

    test_dataset = test_reshape.astype(np.float32)
    test_labels = (np.arange(num_labels)==test_labels[:,None]).astype(np.float32)
    del test_reshape

    print 'Dataset\t\tFeatureShape\t\tLabelShape'
    print 'Training set:\t', train_dataset.shape,'\t', train_labels.shape
    print 'Validation set:\t', valid_dataset.shape, '\t', valid_labels.shape
    print 'Testing set:\t', test_dataset.shape, '\t', test_labels.shape

    # Shared parameters
    batch_size = 16
    kernel1 = 3
    kernel2 = 5
    num_filter = 256
    fc_size1 = 128

    # The Simple Linearly Cascade Network
    graph = tf.Graph()

    with graph.as_default():
        # Input Data
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)
        
        # Setup Variables
        conv1_wt = tf.Variable(tf.truncated_normal([kernel2, kernel2, num_channels, num_filter], stddev=.1))
        conv1_bi = tf.Variable(tf.zeros(shape=[num_filter]))
        
        conv2_wt = tf.Variable(tf.truncated_normal([kernel1, kernel1, num_filter, num_filter], stddev=.1))
        conv2_bi = tf.Variable(tf.constant(1.0, shape=[num_filter]))
        
        conv3_wt = tf.Variable(tf.truncated_normal([kernel2, kernel2, num_filter, num_filter], stddev=.1))
        conv3_bi = tf.Variable(tf.constant(1.0, shape=[num_filter]))
        
        fc1_wt = tf.Variable(tf.truncated_normal([image_size/4*image_size/4*num_filter, fc_size1], stddev=.1))
        fc1_bi = tf.Variable(tf.constant(1.0, shape=[fc_size1]))
        
        fc2_wt = tf.Variable(tf.truncated_normal([fc_size1, num_labels], stddev=.1))
        fc2_bi = tf.Variable(tf.zeros([num_labels]))
        
        # Parameters
        keep_prob = tf.placeholder(tf.float32)
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(.0001, global_step, 1000, .7, staircase=True)
        
        # Model
        def model(data, train=True, keep_prob=.7):
            conv = tf.nn.conv2d(data, conv1_wt, [1,1,1,1], padding='SAME')
            relu = tf.nn.relu(conv + conv1_bi)
            pool = tf.nn.max_pool(relu, [1,2,2,1], [1,2,2,1], padding='SAME')
            
            conv = tf.nn.conv2d(pool, conv2_wt, [1,1,1,1], padding='SAME')
            relu = tf.nn.relu(conv + conv2_bi)
            
            conv = tf.nn.conv2d(relu, conv3_wt, [1,1,1,1], padding='SAME')
            relu = tf.nn.relu(conv + conv3_bi)
            
            pool = tf.nn.max_pool(relu, [1,2,2,1], [1,2,2,1], padding='SAME')
            
            shape = pool.get_shape().as_list()
            reshape = tf.reshape(pool, [shape[0], shape[1]*shape[2]*shape[3]])
            fc = tf.nn.relu(tf.matmul(reshape, fc1_wt) + fc1_bi)
            if train:
                fc = tf.nn.dropout(fc, keep_prob)
            return tf.matmul(fc, fc2_wt) + fc2_bi
        
        # Compute Loss Function
        logits = model(tf_train_dataset, True, keep_prob)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
        
        # Optimizer
        optimizer = tf.train.GradientDescentOptimizer(.0001).minimize(loss)
        
        # Prediction
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(model(tf_valid_dataset, False))
        test_prediction = tf.nn.softmax(model(tf_test_dataset, False))

    num_steps = 1001
    loss_val_cd = np.zeros(num_steps)
    train_acc_cd = np.zeros(num_steps)
    valid_acc_cd = np.zeros(num_steps)

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob : .7}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            
            loss_val_cd[step] = l
            train_acc_cd[step] = accuracy(predictions, batch_labels)
            valid_acc_cd[step] = accuracy(valid_prediction.eval(), valid_labels)
            if (step % 5 == 0):
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % train_acc_cd[step])
                print('Validation accuracy: %.1f%%' % valid_acc_cd[step])
        print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
