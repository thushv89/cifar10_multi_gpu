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

batch_size = 128

def initialize_cnn():
    param_list = []

    logger.info('Initializing the CNN...\n')

    with tf.variable_scope(TF_CONV1_STR) as scope:
        param_list.append(tf.get_variable(
            name=TF_WEIGHTS_STR,initializer=tf.truncated_normal([5,5,3,64], stddev=0.02),dtype=tf.float32)
        )
        param_list.append(
            tf.get_variable(name=TF_BIAS_STR,initializer=tf.zeros(shape=[64],dtype=tf.float32),dtype=tf.float32)
        )


    with tf.variable_scope(TF_CONV2_STR) as scope:
        param_list.append(tf.get_variable(
            name=TF_WEIGHTS_STR, initializer=tf.truncated_normal([5, 5, 64, 128], stddev=0.02), dtype=tf.float32)
        )
        param_list.append(
            tf.get_variable(name=TF_BIAS_STR, initializer=tf.zeros(shape=[128], dtype=tf.float32), dtype=tf.float32)
        )

    with tf.variable_scope(TF_LOCAL1_STR) as scope:
        param_list.append(tf.get_variable(
            name=TF_WEIGHTS_STR,initializer=tf.truncated_normal([cnn_hyps[op]['in'], 512],stddev=0.01),
            dtype=tf.float32)
        )
        param_list.append(tf.get_variable(
            initializer=tf.constant(np.random.random() * 0.001, shape=[512]),
            name=TF_BIAS_STR, dtype=tf.float32)
        )

    with tf.variable_scope(TF_LOCAL2_STR) as scope:
        param_list.append(tf.get_variable(
            name=TF_WEIGHTS_STR,initializer=tf.truncated_normal([512, 256],stddev=0.01),
            dtype=tf.float32)
        )
        param_list.append(tf.get_variable(
            initializer=tf.constant(np.random.random() * 0.001, shape=[256]),
            name=TF_BIAS_STR, dtype=tf.float32)
        )

    with tf.variable_scope(TF_SOFTMAX_STR) as scope:
        param_list.append(tf.get_variable(
            name=TF_WEIGHTS_STR, initializer=tf.truncated_normal([256, 10], stddev=0.01),
            dtype=tf.float32)
        )
        param_list.append(tf.get_variable(
            initializer=tf.constant(np.random.random() * 0.001, shape=[10]),
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
        logger.debug('\tConvolving (%s) With Weights:%s Stride:%s'%(op,cnn_hyperparameters[op]['weights'],cnn_hyperparameters[op]['stride']))
        x = tf.nn.conv2d(x, w, [1,1,1,1], padding='SAME')
        x = tf.nn.relu(x + b)
        #logger.debug('\t\tX after %s:%s'%tf.shape(weights[op]).eval())

        logger.debug('\tPooling (%s) with Kernel:%s Stride:%s'%(op,cnn_hyperparameters[op]['kernel'],cnn_hyperparameters[op]['stride']))
        x = tf.nn.max_pool(x,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
        #logger.debug('\t\tX after %s:%s'%(op,tf.shape(x).eval()))

    with tf.variable_scope(TF_CONV2_STR):
        w, b = tf.get_variable(TF_WEIGHTS_STR), tf.get_variable(TF_BIAS_STR)
        logger.debug('\tConvolving (%s) With Weights:%s Stride:%s' % (
        op, cnn_hyperparameters[op]['weights'], cnn_hyperparameters[op]['stride']))
        x = tf.nn.conv2d(x, w, [1, 1, 1, 1], padding='SAME')
        x = tf.nn.relu(x + b)
        # logger.debug('\t\tX after %s:%s'%tf.shape(weights[op]).eval())

        logger.debug('\tPooling (%s) with Kernel:%s Stride:%s' % (
        op, cnn_hyperparameters[op]['kernel'], cnn_hyperparameters[op]['stride']))
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope(TF_LOCAL1_STR):
        w, b = tf.get_variable(TF_WEIGHTS_STR), tf.get_variable(TF_BIAS_STR)
        # we need to reshape the output of last subsampling layer to
        # convert 4D output to a 2D input to the hidden layer
        # e.g subsample layer output [batch_size,width,height,depth] -> [batch_size,width*height*depth]
        logger.debug('Input size of fulcon_out : %d', cnn_hyperparameters[op]['in'])
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


def tower_loss(logits,labels,weighted=False,tf_data_weights=None):
    # Training computation.
    if include_l2_loss:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels)) + \
               (beta/2)*tf.reduce_sum([tf.nn.l2_loss(w) if 'fulcon' in kw or 'conv' in kw else 0 for kw,w in weights.items()])
    else:
        # use weighted loss
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))

    return loss


def optimize_with_momenutm_func(loss, global_step, learning_rate):
    global tf_weight_vels,tf_bias_vels
    vel_update_ops, optimize_ops = [],[]

    if research_parameters['adapt_structure'] or research_parameters['use_custom_momentum_opt']:
        # custom momentum optimizing
        # apply_gradient([g,v]) does the following v -= eta*g
        # eta is learning_rate
        # Since what we need is
        # v(t+1) = mu*v(t) - eta*g
        # theta(t+1) = theta(t) + v(t+1) --- (2)
        # we form (2) in the form v(t+1) = mu*v(t) + eta*g
        # theta(t+1) = theta(t) - v(t+1)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

        for op in tf_weight_vels.keys():
            [(grads_w,w),(grads_b,b)] = optimizer.compute_gradients(loss, [weights[op], biases[op]])

            # update velocity vector
            vel_update_ops.append(tf.assign(tf_weight_vels[op], research_parameters['momentum']*tf_weight_vels[op] + grads_w))
            vel_update_ops.append(tf.assign(tf_bias_vels[op], research_parameters['momentum']*tf_bias_vels[op] + grads_b))

            optimize_ops.append(optimizer.apply_gradients(
                        [(tf_weight_vels[op]*learning_rate,weights[op]),(tf_bias_vels[op]*learning_rate,biases[op])]
            ))
    else:
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

