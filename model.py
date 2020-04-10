import tensorflow as tf

def lenet(x, is_training):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    net = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, 2, 2)
    net = tf.layers.conv2d(net, 64, 3, activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, 2, 2)
    net = tf.contrib.layers.flatten(net)
    net = tf.layers.dense(net, 1024)
    net = tf.layers.dropout(net, rate=0.4, training=is_training)
    return tf.layers.dense(net, 10)

def model_fn(features, labels, mode, params):
    predict = lenet(features, mode == mode)
    predicted_classes = tf.argmax(predict, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(predict),
            'logits': predict,
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=predict)
    tf.summary.scalar('loss', loss)
    accuracy = tf.metrics.accuracy(predictions=predicted_classes, labels=labels)
    tf.summary.scalar('accuracy', accuracy[1])
    eval_metric_ops = {"accuracy": accuracy}
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=eval_metric_ops)
    
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=params["learning_rate"])
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    train_hook_list= []
    train_tensors_log = {'accuracy': accuracy[1],
                         'loss': loss,
                         'global_step': tf.train.get_global_step()}
    train_hook_list.append(tf.train.LoggingTensorHook(
        tensors=train_tensors_log, every_n_iter=100))
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        training_hooks=train_hook_list)