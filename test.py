import tensorflow as tf
import numpy as np
import sklearn.metrics as sk
import sklearn.preprocessing as pp
import utils


# TF config, allowing to use GPU by multiple programs
config = tf.ConfigProto()  
config.gpu_options.per_process_gpu_memory_fraction = 0.5 
config.gpu_options.allow_growth = True     
config.allow_soft_placement = True

test_file = 'path/to/test/set/test.libsvm'
model_file = 'path/to/model/file/model.ckpt'
filter_number = 64
feature_num = 20000
gpu_num = 2


graph = tf.Graph()

# Create the model: 2 layer FC NN, same as training
with graph.as_default():
    sp_indice = tf.placeholder(tf.int64)
    sp_value = tf.placeholder(tf.float32)
    x =  tf.SparseTensor(sp_indice, sp_value, [batch_size, feature_num])
    y_ = tf.placeholder(tf.float32, [None, 2])
    keep_prob = tf.placeholder("float32")
    W_fc1 = utils.weight_variable([feature_num, filter_number])
    b_fc1 = utils.bias_variable([filter_number])
    W_fc2 = utils.weight_variable([filter_number, filter_number])
    b_fc2 = utils.bias_variable([filter_number])
    W_out = utils.weight_variable([filter_number, 2])
    b_out = utils.bias_variable([2]) 
    tower_pred = [[] for _ in range(gpu_num)]
    tower_loss = [0.0 for _ in range(gpu_num)]
    for i in range(gpu_num):
        with tf.device("/gpu:%d" % i):
            # We split the data into $gpu_num parts.
            next_batch = tf.sparse_split(split_dim=0, num_split=gpu_num, sp_input=x)[i]
            next_label = y_[i * batch_size / gpu_num: (i+1) * batch_size / gpu_num, :]
            hidden_1 = tf.nn.relu(tf.sparse_tensor_dense_matmul(next_batch, W_fc1) + b_fc1)
            hidden_2 = tf.nn.dropout(tf.nn.relu(tf.matmul(hidden_1, W_fc2) + b_fc2), keep_prob)
            tower_pred[i] = tf.nn.softmax(tf.matmul(hidden_2, W_out) + b_out)
            tower_loss[i] = -tf.reduce_mean(tf.reduce_sum(tf.cast(next_label, "float") * tf.log(tf.clip_by_value(tf.cast(tower_pred[i], "float"), 1e-10, 1.0)), reduction_indices=[1]))   
            params = tf.trainable_variables()
            tf.get_variable_scope().reuse_variables()
    pred = tf.concat(0, [tower_pred[_] for _ in range(gpu_num)])
    loss = tower_loss[0]
    for _ in range(1, gpu_num):
        loss = tf.add(loss, tower_loss[_])
    loss = loss / (gpu_num + 0.0)


with graph.as_default():
    with tf.Session(config=config, graph=graph) as sess:
        saver = tf.train.Saver()
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, model_file)
        label_v, ids_v, values_v = utils.libsvm_data_read(test_file)
        test_num = int(label_v.shape[0] / batch_size)
        iter = 1
        test_ids = ids_v[0: batch_size]
        test_values = values_v[0: batch_size]
        test_labels = label_v[0: batch_size]
        ids_flatten_v, value_flatten_v = utils.libsvm_convert_sparse_tensor(test_ids, test_values)
        y_prediction = sess.run(pred, feed_dict={sp_indice: ids_flatten_v , sp_value: value_flatten_v, y_: test_labels, keep_prob: 1.0})
       	test_y_real = test_labels
        for index in range(1, test_num):
            print 'Testing Stage ' + str(index) + ' / ' + str(test_num)
            test_ids = ids_v[index * batch_size: (index + 1) * batch_size]
            test_values = values_v[index * batch_size: (index + 1) * batch_size]
            test_labels = label_v[index * batch_size:(index + 1) * batch_size]
            ids_flatten_v, value_flatten_v = utils.libsvm_convert_sparse_tensor(test_ids, test_values)
            test_y_real = np.concatenate((test_y_real, test_labels), 0)
            y_prediction = np.concatenate((y_prediction, sess.run(pred, feed_dict={sp_indice: ids_flatten_v, sp_value: value_flatten_v, y_: test_labels, keep_prob: 1.0})), 0)
        test_y_real = np.argmax(test_y_real, 1)
        for threshold in range(0, 1001, 1):
            test_prediction = np.array(pp.binarize(np.array([y_prediction[:, 1]]), threshold / 1000.0)[0])
            cm = sk.confusion_matrix(test_y_real, test_prediction)
            print cm
            # The former two values are used for ROC, the later two are used for P-R
            print str(cm[0][1] / (cm[0][1] + cm[0][0] + 0.0)) + '\t' + str(cm[1][1] / (cm[1][1] + cm[1][0] + 0.0)) + \
            '\t' + str(cm[1][1] / (cm[1][1] + cm[1][0] + 0.0)) + '\t' + str(cm[1][1] / (cm[1][1] + cm[0][1] + 0.0))
