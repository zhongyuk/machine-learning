import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from sklearn.cross_validation import train_test_split

def unpickle(file):
    # Load pickled data
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict

def accuracy(pred, labels):
    # Compute accuracy
    return (1.0 * np.sum(np.argmax(pred, 1) == np.argmax(labels, 1))/ pred.shape[0])

def count_correct(pred, labels):
    # Count number of correctly classified samples
    return np.sum(np.argmax(pred, 1) == np.argmax(labels, 1))

def load_data():
    # Load training and testing data
    train_fnroot = 'cifar-10-batches-py/data_batch_'
    test_filename = 'cifar-10-batches-py/test_batch'
    train_dataset = None
    test_dataset = None
    print "Loading the training data..."
    for i in range(1,6):
	train_filename = train_fnroot + str(i)
	batch = unpickle(train_filename)
	if i==1:
	    train_dataset = batch['data']
	    train_labels = np.array(batch['labels'])
	else:
	    train_dataset = np.concatenate((train_dataset, batch['data']), axis=0)
	    train_labels = np.concatenate((train_labels, batch['labels']))
    print "Loading the testing data..."
    test_batch = unpickle(test_filename)
    test_dataset = test_batch['data']
    test_labels = np.array(test_batch['labels'])
    return train_dataset, train_labels,  test_dataset, test_labels
  
def augment_data():
    # 1) Mirror Reflection
    # 2) Random Corp
    # 3) Color Jitter
    pass

def preprocess_data(data_matrix):
    # 1) Center the data/Subtract Mean 2) Change datatype
    pass

def initialize_variable(shape, mean=0.0, std=.1):
    # Initialize weights and biases based on given shape
    wt = tf.Variable(tf.truncated_normal(shape=shape, mean=mean, stddev=std ))
    bi = tf.Variable(tf.random_normal(shape=[shape[-1]], mean=mean, stddev=std))
    return wt, bi

def conv_layer(x, w, b, stride=1, padding='SAME'):
    # Perform a convolution layer computation followed by a ReLu activation
    # padding: "SAME" or "VALID" 
    conv = tf.nn.conv2d(x, w, [1, stride, stride, 1], padding=padding)
    relu = tf.nn.relu(conv + b)
    return relu

def pool_layer(x, method='max', kernel=2, stride=2, padding='SAME'):
    # Perform a down sampling layer computation - "max" : max pooling, "avg" : avg pooling
    if method=="max":
	return tf.nn.max_pool(x, [1,kernel,kernel,1], [1,stride,stride,1], padding=padding)
    elif method=='avg':
	return tf.nn.avg_pool(x, [1,kernel,kernel,1], [1,stride,stride,1], padding=padding)
    else:
	raise ValueError

def full_layer(x, w, b, dropout=True, keep_prob=.5):
    # Perform a fully connected layer computation followed by a ReLu activation
    # If dropout is True, drop out is performed
    fc = tf.nn.relu(tf.matmul(x,w) + b)
    if dropout:
	fc = tf.nn.dropout(fc, keep_prob)
    return fc

def convnet_model(data, weights, baises, dropout=True, keep_prob=.5):
    # Construct convolution layers
    conv = conv_layer(data, weights['conv1_wt'], biases['conv1_bi'])
    pool = pool_layer(conv, 'max')
    conv = conv_layer(pool, weights['conv2_wt'], biases['conv2_bi'], padding="SAME")
    pool = pool_layer(conv, 'max')
    conv = conv_layer(pool, weights['conv3_wt'], biases['conv3_bi'])
    pool = pool_layer(conv, 'max')
    # Reshape data from 4D into 2D, prepare for fully connected layers
    shape = pool.get_shape().as_list()
    data = tf.reshape(pool, [shape[0], shape[1]*shape[2]*shape[3]])
    # Construct fully connected layers
    if dropout:
        fc = full_layer(data, weights['fc1_wt'], biases['fc1_bi'], True, keep_prob)
    #fc = full_layer(fc, weights['fc2_wt'], biases['fc2_bi'], True, keep_prob)
    else:
        fc = full_layer(data, weights['fc1_wt'], biases['fc1_bi'], False)
    #fc = full_layer(fc, weights['fc2_wt'], biases['fc2_bi'], False)
    output = full_layer(fc, weights['fc2_wt'], biases['fc2_bi'], False)
    return output


def setup_graph(tf_data, convnet_shapes, hyperparams,):
    # Setup ConvNet Computation Graph
    print "Prepare network parameters", "."*32
    graph = tf.Graph()
    
    with graph.as_default():
        # Setup training, validation, testing dataset
        tf_train_dataset = tf_data['train_X']
        tf_train_labels = tf_data['train_y']
        tf_valid_dataset = tf_data['valid_X']
        tf_valid_lables = tf_data['valid_y']
        tf_test_dataset = tf_data['test_X']
        tf_test_labels = tf_data['test_y']
        
        # Initialize weights and biases
        weights, biases = {}, {}
        for key in convnet_shapes.keys():
            weights[key], biases[key] = initialize_variable(convnet_shapes[key], \
            hyperparams['init_mean'], hyperparams['init_std'])
        
        # HyperParameters
        keep_prob = hyperparams['keep_prob']
        learning_rate = hyperparams['learning_rate']
        
        # Compute Loss Function
        train_logits = convnet_model(tf_train_dataset, weights, biases, True, keep_prob)
        train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(train_logits, tf_train_labels))
        
        # Optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(train_loss)
        
        # Prediction
        train_prediction = tf.nn.softmax(train_logits)
        valid_logits = convnet_model(tf_valid_dataset, weights, biases, False`)
        valid_prediction = tf.nn.softmax(valid_logits)
        valid_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(valid_logits, tf_valid_lables))
        if not tf_test_dataset:
            test_prediction = tf.nn.softmax(convnet_model(tf_test_dataset, weights, biases, False))
        else:
            test_prediction = None
    return graph

def train_model(graph, steps, feed_dict={})
    # Train Convnet
    print "Start training", '.'*32
    num_steps = steps
    train_losses = np.zeros(num_steps)
    valid_losses = np.zeros(num_steps)
    train_acc = np.zeros(num_steps)
    valid_acc = np.zeros(num_steps)
    
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        for step in range(num_steps):
            _, tl, predictions = session.run([optimizer, train_loss, train_prediction], feed_dict=feed_dict)
            train_losses[step] = tl
            train_acc[step] = accuracy(predictions, train_label_batch)
            # Compute validation set accuracy batch by batch
            valid_losses[step] = valid_loss.eval()
            valid_acc[step] = accuracy(valid_prediction.eval(), tf_valid_labels.eval())
            if ((step % 50 == 0) or (step<20)):
                print('Training loss at step %d: %f' % (step, tl))
                print('Training accuracy: %.1f%%' % (train_acc[step]*100))
                print('Validation loss at step %d: %f' % (step, valid_losses[step]))
                print('Validation accuracy: %.1f%%' % (valid_acc[step]*100))
    # Compute test set accuracy
    print "Finished training", '.'*32
    if not test_prediction:
        test_acc = accuracy(test_prediction.eval(), tf_teset_labels.eval())
    else:
        test_acc = None
    # Group training data into a dictionary
    training_data = {'train_losses' : train_losses, 'train_acc' : train_acc, \
                     'valid_losses' : valid_losses, 'valid_acc' : valid_acc, 'test_acc' : test_acc}
    return graph, training_data



if __name__=='__main__':
    # Load Data
    print "Load data", "."*32
    train_dataset, train_labels, test_dataset, test_labels = load_data()

    # Split 20% of training set as validation set
    print "Split training and validation set", "."*32    
    train_dataset, valid_dataset, train_labels, valid_labels = \
    train_test_split(train_dataset, train_labels, test_size=10000,\
    random_state=897, stratify=train_labels)

    # Print out data shapes
    print 'Dataset\t\tFeatureShape\tLabelShape'
    print 'Training set:\t', train_dataset.shape,'\t', train_labels.shape
    print 'Validation set:\t', valid_dataset.shape,'\t', valid_labels.shape
    print 'Testing set:\t', test_dataset.shape, '\t', test_labels.shape

    # Reshape the data into pixel by pixel by RGB channels
    print "Reformat data", "."*32
    train_dataset = np.rollaxis(train_dataset.reshape((-1,3,32,32)), 1, 4)
    valid_dataset = np.rollaxis(valid_dataset.reshape((-1,3,32,32)), 1, 4)
    test_dataset = np.rollaxis(test_dataset.reshape((-1,3,32,32)), 1, 4)
    print 'Dataset\t\tFeatureShape\t\tLabelShape'
    print 'Training set:\t', train_dataset.shape,'\t', train_labels.shape
    print 'Validation set:\t', valid_dataset.shape, '\t', valid_labels.shape
    print 'Testing set:\t', test_dataset.shape, '\t', test_labels.shape

    # Mirror Reflection
    #train_LRF = train_reshape[:,:,::-1,:]

    # Data Preprocess: change datatype; center the data
    print "Preprocess data", "."*32
    train_dataset = train_dataset.astype(np.float32)
    train_labels = (np.arange(num_labels)==train_labels[:,None]).astype(np.float32)

    valid_dataset = valid_dataset.astype(np.float32)
    valid_labels = (np.arange(num_labels)==valid_labels[:, None]).astype(np.float32)

    test_dataset = test_dataset.astype(np.float32)
    test_labels = (np.arange(num_labels)==test_labels[:,None]).astype(np.float32)
    print 'Dataset\t\tFeatureShape\t\tLabelShape'
    print 'Training set:\t', train_dataset.shape,'\t', train_labels.shape
    print 'Validation set:\t', valid_dataset.shape, '\t', valid_labels.shape
    print 'Testing set:\t', test_dataset.shape, '\t', test_labels.shape
    
    # Dataset Parameters
    image_size = 32
    num_labels = 10
    num_channels = 3
    
    # Network parameters
    batch_size = 64
    kernel_size3 = 3
    kernel_size5 = 5
    num_filter = 16
    fc_size1 = 256

    # Setup shapes for each layer in the convnet
    convnet_shapes = {'conv1' : [kernel_size5, kernel_size5, num_channels, num_filter],
                      'conv2' : [kernel_size3, kernel_size3, num_filter, num_filter],
                      'conv3' : [kernel_size5, kernel_size5, num_filter, num_filter],
                      'fc1'   : [(image_size/2/2/2)**2*num_filter, fc_size1],
                      'fc2'   : [fc_size1, num_labels]}

    # Prepare small batch of data for experimental training
    train_batch, train_label_batch = train_dataset[:batch_size, :, :, :], train_labels[:batch_size,:]
    valid_batch, valid_label_batch = valid_dataset[:32, :, :, :], valid_labels[:32,:]
    print 'Dataset\t\tFeatureShape\t\tLabelShape'
    print 'Training set:\t', train_dataset.shape,'\t', train_labels.shape

    # Prepare data for tensorflow
    tf_data = {'train_X': tf.constant(train_batch), 'train_y': tf.constant(train_label_batch),
               'valid_X': tf.constant(valid_batch), 'valid_y': tf.constant(valid_label_batch),
               'test_X' : None, 'test_y': None}

    # HyperParameters
    hyperparams = {'keep_prob': 1.0, 'init_mean': 0.0, 'init_std': 0.01, 'learning_rate': 0.01}

    # Setup computation graph
    graph = setup_graph(tf_data, convnet_shapes, hyperparams)

    # Train convnet
    steps = 1001
    graph, training_data = train_model(graph, steps)
    




