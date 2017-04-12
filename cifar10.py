import tensorflow as tf
import numpy as np

logger = None

TF_WEIGHTS_STR = 'weights'
TF_BIAS_STR = 'bias'
TF_CONV1_STR = 'conv1'
TF_CONV2_STR = 'conv2'
TF_LOCAL1_STR = 'local1'
TF_LOCAL2_STR = 'local2'
TF_SOFTMAX_STR = 'softmax'

image_size = 24
batch_size = 128
learning_rate = 0.01
dropout = (True,0.5)

CONV1_SHAPE = [5,5,3,64]
CONV2_SHAPE = [5,5,64,128]
CONV_STRIDE = [1,1,1,1]
POOL1_SHAPE = [1,3,3,1]
POOL2_SHAPE = [1,3,3,1]
POOL_STRIDE = [1,2,2,1]

def get_last_2d_output_size():
    '''
    This method calculates the size of the final 2D output
    just before the fully connected layers
    This assumes that padding type is 'SAME' for convolution and pooling
    and assumes size = width = height (that inputs are square)
    :return: size of the last 2d output
    '''
    last_output_size = image_size
    last_output_size = int(last_output_size/CONV_STRIDE[1]) # stride from conv1
    last_output_size = int(last_output_size/POOL_STRIDE[1]) # stride by pool1
    last_output_size = int(last_output_size/CONV_STRIDE[1]) # stride from conv2
    last_output_size = int(last_output_size/POOL_STRIDE[1]) # stride by pool2

    return last_output_size

FC1_IN = get_last_2d_output_size()**2 * CONV2_SHAPE[-1]
FC1_OUT = 512
FC2_IN = 512
FC2_OUT = 256
OUT_CLASSES = 10

def initialize_cnn():
    param_list = []

    logger.info('Initializing the CNN...\n')

    with tf.variable_scope(TF_CONV1_STR) as scope:
        param_list.append(tf.get_variable(
            name=TF_WEIGHTS_STR,initializer=tf.truncated_normal(CONV1_SHAPE, stddev=0.02),dtype=tf.float32)
        )
        param_list.append(
            tf.get_variable(name=TF_BIAS_STR,initializer=tf.zeros(shape=CONV1_SHAPE[-1],dtype=tf.float32),dtype=tf.float32)
        )

    with tf.variable_scope(TF_CONV2_STR) as scope:
        param_list.append(tf.get_variable(
            name=TF_WEIGHTS_STR, initializer=tf.truncated_normal(CONV2_SHAPE, stddev=0.02), dtype=tf.float32)
        )
        param_list.append(
            tf.get_variable(name=TF_BIAS_STR, initializer=tf.zeros(shape=CONV2_SHAPE[-1], dtype=tf.float32), dtype=tf.float32)
        )

    with tf.variable_scope(TF_LOCAL1_STR) as scope:
        param_list.append(tf.get_variable(
            name=TF_WEIGHTS_STR,initializer=tf.truncated_normal([FC1_IN, FC1_OUT],stddev=0.01),
            dtype=tf.float32)
        )
        param_list.append(tf.get_variable(
            initializer=tf.constant(np.random.random() * 0.001, shape=[FC1_OUT]),
            name=TF_BIAS_STR, dtype=tf.float32)
        )

    with tf.variable_scope(TF_LOCAL2_STR) as scope:
        param_list.append(tf.get_variable(
            name=TF_WEIGHTS_STR,initializer=tf.truncated_normal([FC2_IN, FC2_OUT],stddev=0.01),
            dtype=tf.float32)
        )
        param_list.append(tf.get_variable(
            initializer=tf.constant(np.random.random() * 0.001, shape=[FC2_OUT]),
            name=TF_BIAS_STR, dtype=tf.float32)
        )

    with tf.variable_scope(TF_SOFTMAX_STR) as scope:
        param_list.append(tf.get_variable(
            name=TF_WEIGHTS_STR, initializer=tf.truncated_normal([FC2_OUT, OUT_CLASSES], stddev=0.01),
            dtype=tf.float32)
        )
        param_list.append(tf.get_variable(
            initializer=tf.constant(np.random.random() * 0.001, shape=[OUT_CLASSES]),
            name=TF_BIAS_STR, dtype=tf.float32)
        )

    return param_list


def inference(dataset):
    logger.debug('Defining the logit calculation ...')

    x = dataset
    logger.debug('\tReceived data for X(%s)...'%x.get_shape().as_list())

    #need to calculate the output according to the layers we have
    with tf.variable_scope(TF_CONV1_STR):
        w,b = tf.get_variable(TF_WEIGHTS_STR),tf.get_variable(TF_BIAS_STR)
        logger.debug('\tConvolving with Weights:%s Stride:%s'%(CONV1_SHAPE,CONV_STRIDE))
        x = tf.nn.conv2d(x, w, [1,1,1,1], padding='SAME')
        x = tf.nn.relu(x + b)
        #logger.debug('\t\tX after %s:%s'%tf.shape(weights[op]).eval())

        logger.debug('\tPooling with Kernel:%s Stride:%s'%(POOL1_SHAPE,POOL_STRIDE))
        x = tf.nn.max_pool(x,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
        #logger.debug('\t\tX after %s:%s'%(op,tf.shape(x).eval()))

    with tf.variable_scope(TF_CONV2_STR):
        w, b = tf.get_variable(TF_WEIGHTS_STR), tf.get_variable(TF_BIAS_STR)
        logger.debug('\tConvolving with Weights:%s Stride:%s' % (CONV2_SHAPE,CONV_STRIDE))
        x = tf.nn.conv2d(x, w, [1, 1, 1, 1], padding='SAME')
        x = tf.nn.relu(x + b)
        # logger.debug('\t\tX after %s:%s'%tf.shape(weights[op]).eval())

        logger.debug('\tPooling with Kernel:%s Stride:%s'%(POOL1_SHAPE,POOL_STRIDE))
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope(TF_LOCAL1_STR):
        w, b = tf.get_variable(TF_WEIGHTS_STR), tf.get_variable(TF_BIAS_STR)
        # we need to reshape the output of last subsampling layer to
        # convert 4D output to a 2D input to the hidden layer
        # e.g subsample layer output [batch_size,width,height,depth] -> [batch_size,width*height*depth]
        logger.debug('Input size of FC1 : %d', FC1_IN)
        input_shape = x.get_shape().aslist()
        x = tf.reshape(x, [batch_size, input_shape[1]*input_shape[2]*input_shape[3]])
        x = tf.nn.relu(tf.matmul(x, w) + b)

    with tf.variable_scope(TF_LOCAL2_STR):
        w,b = tf.get_variable(TF_WEIGHTS_STR),tf.get_variable(TF_BIAS_STR)
        x = tf.nn.relu(tf.matmul(x, w) + b)

    with tf.variable_scope(TF_SOFTMAX_STR):
        w,b = tf.get_variable(TF_WEIGHTS_STR),tf.get_variable(TF_BIAS_STR)
        x = tf.matmul(x, w) + b

    return x


def tower_loss(logits,labels):
    # Computing loss

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels)) + \
               (beta/2)*tf.reduce_sum([tf.nn.l2_loss(w) if 'fulcon' in kw or 'conv' in kw else 0 for kw,w in weights.items()])
    else:
        # use weighted loss
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))

    return loss


def optimize_with_momenutm(optimizer, loss, global_step, learning_rate):

    optimize_ops.append(
            tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                       momentum=research_parameters['momentum']).minimize(loss))

    return optimize_ops,vel_update_ops,learning_rate

def predict_with_logits(logits):
    # Predictions for the training, validation, and test data.
    prediction = tf.nn.softmax(logits)
    return prediction


def predict_with_dataset(dataset):
    logits = inference(dataset)
    prediction = tf.nn.softmax(logits)
    return prediction


def accuracy(predictions, labels):
    assert predictions.shape[0]==labels.shape[0]
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def init_velocity_vectors(cnn_ops,cnn_hyps):
    global weight_velocity_vectors,bias_velocity_vectors

    for op in cnn_ops:
        if 'conv' in op:
            weight_velocity_vectors[op] = tf.zeros(shape=cnn_hyps[op]['weights'],dtype=tf.float32)
            bias_velocity_vectors[op] = tf.zeros(shape=[cnn_hyps[op]['weights'][3]],dtype=tf.float32)
        elif 'fulcon' in op:
            weight_velocity_vectors[op] = tf.zeros(shape=[cnn_hyps[op]['in'],cnn_hyps[op]['out']],dtype=tf.float32)
            bias_velocity_vectors[op] = tf.zeros(shape=[cnn_hyps[op]['out']],dtype=tf.float32)

