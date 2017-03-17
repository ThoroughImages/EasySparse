import tensorflow as tf
import numpy as np


def read_and_decode_batch(filename_queue, batch_size, capacity, min_after_dequeue):
    """Dequeue a batch of data from the TFRecord.
    Args:
    filename_queue: Filename Queue of the TFRecord.
    batch_size: How many records dequeued each time.
    capacity: The capacity of the queue.
    min_after_dequeue: Ensures a minimum amount of shuffling of examples.
    Returns:
     List of the dequeued (batch_label, batch_ids, batch_values).
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    batch_serialized_example = tf.train.shuffle_batch([serialized_example], 
        batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
    # The feature definition here should BE consistent with LibSVM TO TFRecord process.
    features = tf.parse_example(batch_serialized_example,
                                       features={
                                           "label": tf.FixedLenFeature([], tf.float32),
                                           "ids": tf.VarLenFeature(tf.int64),
                                           "values": tf.VarLenFeature(tf.float32)
                                       })
    batch_label = features["label"]
    batch_ids = features["ids"]
    batch_values = features["values"]
    return batch_label, batch_ids, batch_values

    
def sparse_tensor_to_train_batch(dense_label, dense_ids, dense_values):
    """Transform the dence ids and values to TF understandable inputs. Meanwhile, one-hot encode the labels.
    For instance, for dense_ids in the form of
    [[1, 4, 6, -1],
     [2, 3, -1, -1],
     [3, 4, 5, 6], ...
    ]
    should be transformed into
    [[0, 1], [0, 4], [0, 6],
     [1, 2], [1, 3],
     [2, 3], [2, 4], [2, 5], [2, 6], ...
    ]
    For dense_values in the form of:
    [[0.01, 0.23, 0.45, -1],
     [0.34, 0.25, -1, -1],
     [0.23, 0.78, 0.12, 0.56], ...
    ]
    should be transformed into
    [0.01, 0.23, 0.45, 0.34, 0.25, 0.23, 0.78, 0.12, 0.56, ...]
    Args:
    dense_label: Labels.
    dense_ids: Sparse indices.
    dense_values: Sparse values.
    Returns:
     List of the processed (label, ids, values) ready for training inputs.
    """
    indice_flatten = []
    values_flatten = []
    label_flatten = []
    index = 0
    for i in range(0, dense_label.shape[0]):
        if int(float(dense_label[i])) == 0:
            label_flatten.append([1.0, 0.0])
        else:
            label_flatten.append([0.0, 1.0])
        values = list(dense_values)
        indice = list(dense_ids)
        for j in range(0,len(indice[i])):
            if not indice[i][j] == -1:
                indice_flatten.append([index,indice[i][j]])
                values_flatten.append(values[i][j])
            else:
                break
        index += 1           
    return np.array(label_flatten), indice_flatten, values_flatten


def libsvm_data_read(input_filename):
    """Read all the data from the LibSVM file.
    Args:
    input_filename: Filename of the LibSVM.
    Returns:
     List of the acquired (label, ids, values).
    """
    labels = []
    ids_all = [] 
    values_all = [] 
    for line in open(input_filename, 'r'):
        data = line.split(' ')
        if int(float(data[0])) == 0:
            labels.append([1.0, 0.0])
        else: 
            labels.append([0.0, 1.0]) 
        ids = []
        values = [] 
        for fea in data[1:]:
            id, value = fea.split(':')
            ids.append(int(id))
            values.append(float(value))
        ids_all.append(ids)
        values_all.append(values)
    return np.array(labels), np.array(ids_all), np.array(values_all)


def libsvm_convert_sparse_tensor(array_ids, array_values):
    """Transform the contents into TF understandable formats, which is similar to 
       sparse_tensor_to_train_batch().
    Args:
    array_ids: Sparse indices.
    array_values: Sparse values.
    Returns:
     List of the transformed (ids, values).
    """
    indice_flatten_v = []
    values_flatten_v = []
    index = 0
    for i in range(0, array_ids.shape[0]):
        for j in range(0, len(array_ids[i])):
            indice_flatten_v.append([index, array_ids[i][j]])
            values_flatten_v.append(array_values[i][j]) 
        index += 1        
    return indice_flatten_v, values_flatten_v


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads    
