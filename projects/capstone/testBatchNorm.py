import numpy as np
import tensorflow as tf

def initialize_BatchNorm(scope_name, shape):
    with tf.variable_scope(scope_name) as scope:
        gamma = tf.get_variable("gamma", shape[-1], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable("beta", shape[-1], initializer=tf.constant_initializer(.0))
        avg = tf.get_variable("avg", shape[-1], initializer=tf.constant_initializer(0.0), trainable=False)
        var = tf.get_variable("var", shape[-1], initializer=tf.constant_initializer(1.0), trainable=False)
        moving_avg = tf.get_variable("moving_avg", shape[-1], initializer=tf.constant_initializer(0.0), trainable=False)
        moving_var = tf.get_variable("moving_var", shape[-1], initializer=tf.constant_initializer(1.0), trainable=False)
        scope.reuse_variables()

def BatchNorm(x, scope, train, epsilon=0.001, decay=.9):
    with tf.variable_scope(scope, reuse=True):
        gamma, beta = tf.get_variable("gamma"), tf.get_variable("beta")
        avg, var = tf.get_variable("avg"), tf.get_variable("var")
        moving_avg, moving_var = tf.get_variable("moving_avg"), tf.get_variable("moving_var")
        shape = x.get_shape().as_list()
        if train:
            ema = tf.train.ExponentialMovingAverage(decay=decay)
            batch_avg, batch_var = tf.nn.moments(x, range(len(shape)-1))
            print "before assign............"
            print "avg: ", avg.eval()
            print "moving_avg: ", moving_avg.eval()
            tf.assign(avg, batch_avg)
            tf.assign(var, batch_var)
            # maintain moving averages
            ema_op = ema.apply([avg, var])
            with tf.control_dependencies([ema_op]):
                tf.assign(moving_avg ,ema.average(avg))
                tf.assign(moving_var, ema.average(var))
                print "after assign............"
                print "avg: ", avg.eval()
                print "moving_avg: ", moving_avg.eval()
                outputs = tf.nn.batch_normalization(x, batch_avg, batch_var, beta, gamma, epsilon)
        else:
            outputs = tf.nn.batch_normalization(x, moving_avg, moving_var, beta, gamma, epsilon)
    return outputs

def test_BatchNorm():
    graph = tf.Graph()
    with graph.as_default():
        x = tf.constant(np.array([[0,10],[20,30]],np.float32))
        scope = 'BatchNorm'
        shape = [2,2]
        initialize_BatchNorm(scope, shape)
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        for i in range(5):
            print "iteration ", i, "...................."
            x = BatchNorm(x, scope, True)
            x_valid = BatchNorm(x, scope, False)


if __name__=='__main__':
    test_BatchNorm()

