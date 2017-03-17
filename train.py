import os
import tensorflow as tf
import numpy as np
import utils


# TF config, allowing to use GPU by multiple programs
config = tf.ConfigProto()  
config.gpu_options.per_process_gpu_memory_fraction = 0.5 
config.gpu_options.allow_growth = True     
config.allow_soft_placement = True

# Parameter Definition
training_sample = os.path.join('/path/to/tfrecord', 'train.tfrecords')
validation_sample = '/path/to/validation/set/validation.libsvm'
model_path = './models/fcnn'
filter_number = 64
training_iteration = 10000
batch_size = 2048
display_step = 100
test_step = 1000
save_step = 1000
shuffle_index = 0
starter_learning_rate = 1e-3
feature_num = 20000 # Should be the same as the sparse input shape
epoch_num = None
shuffle_or_not = True
gpu_num = 2
max_model_num_to_keep = 1000
capacity = 40960
min_after_dequeue = 10240
learning_rate_decay_step = 1000
learning_rate_decay_rate = 0.5

# Define TF graph
graph = tf.Graph()

# Training sample queue
with graph.as_default(): 
    filename_queue = tf.train.string_input_producer([training_sample], num_epochs=epoch_num, shuffle=shuffle_or_not)
    label_batch, ids_batch, values_batch = utils.read_and_decode_batch(filename_queue, batch_size, capacity, min_after_dequeue)
    dense_values = tf.sparse_tensor_to_dense(values_batch, -1) # For further process
    dense_ids = tf.sparse_tensor_to_dense(ids_batch, -1) # For further process

# Create the model: 2 layer FC NN
with graph.as_default():
    global_step = tf.Variable(0, trainable=False)
    # Here we use the indices and values to reproduce the input SparseTensor
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
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, learning_rate_decay_step, learning_rate_decay_rate, staircase=True)
    opt_= tf.train.AdamOptimizer(learning_rate)
    tower_grads = []
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
            grads = opt_.compute_gradients(tower_loss[i], var_list = params)
            tower_grads.append(grads)
    pred = tf.concat(0, [tower_pred[_] for _ in range(gpu_num)])
    loss = tower_loss[0]
    for _ in range(1, gpu_num):
        loss = tf.add(loss, tower_loss[_])
    loss = loss / (gpu_num + 0.0)
    grads_ave = utils.average_gradients(tower_grads)
    train_step = opt_.apply_gradients(grads_ave, global_step=global_step)

with graph.as_default():
    with tf.Session(config=config, graph=graph) as sess:
        saver = tf.train.Saver(max_to_keep=max_model_num_to_keep)
        sess.run(tf.initialize_all_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        print 'Reading From LibSVM file...'
        label_v, ids_v, values_v = utils.libsvm_data_read(validation_sample)
        test_num = int(label_v.shape[0] / batch_size)
        iter = 1
        print 'Starting Training Process...'
        try:
            while not coord.should_stop():
                label, ids, values = sess.run([label_batch, dense_ids, dense_values])
                label, ids_flatten, value_flatten = utils.sparse_tensor_to_train_batch(label, ids, values)
                sess.run(train_step, feed_dict={sp_indice: ids_flatten, sp_value: value_flatten, y_: label, keep_prob: 0.5})
                if iter % display_step == 0:
                    print "Iteration:", '%04d' % (iter), ", Training Sample Loss: ", "{:.9f}".format(
                        sess.run(loss, feed_dict={sp_indice: ids_flatten, sp_value: value_flatten, y_: label, keep_prob: 1.0}))
                if iter % test_step == 0:
                    test_loss = 0.0
                    for index in range(0, test_num):
                        test_ids = ids_v[index * batch_size: (index + 1) * batch_size]
                        test_values = values_v[index * batch_size: (index + 1) * batch_size]
                        ids_flatten_v, value_flatten_v = utils.libsvm_convert_sparse_tensor(test_ids,test_values)
                        test_labels = label_v[index * batch_size:(index + 1) * batch_size]
                        test_loss += sess.run(loss, feed_dict={sp_indice: ids_flatten_v , sp_value: value_flatten_v ,y_: test_labels, keep_prob: 1.0})
                    test_loss /= (test_num + 0.0)
                    print 'Validation Loss: %s' % str(test_loss)
                if iter % save_step == 0:
                    save_path = saver.save(sess, model_path + '-' + str(iter) + '.ckpt')   
                iter += 1
        except tf.errors.OutOfRangeError:
            print("Training Done!")
        finally:
            coord.request_stop()
        coord.join(threads)
