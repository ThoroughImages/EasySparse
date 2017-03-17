import os
import numpy as np
import tensorflow as tf


def convert_tfrecords(input_filename, output_filename):
    """Concert the LibSVM contents to TFRecord.
    Args:
    input_filename: LibSVM filename.
    output_filename: Desired TFRecord filename.
    """
    print("Starting to convert {} to {}...".format(input_filename, output_filename))
    writer = tf.python_io.TFRecordWriter(output_filename)

    for line in open(input_filename, "r"):
        data = line.split(" ")
        label = float(data[0])
        ids = [] 
        values = [] 
        for fea in data[1:]:
            id, value = fea.split(":")
            ids.append(int(id))
            values.append(float(value))
        # Write samples one by one
        example = tf.train.Example(features=tf.train.Features(feature={
            "label":
                tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
            "ids":
                tf.train.Feature(int64_list=tf.train.Int64List(value=ids)),
            "values":
                tf.train.Feature(float_list=tf.train.FloatList(value=values))
        }))
        writer.write(example.SerializeToString())
    writer.close()
    print("Successfully converted {} to {}!".format(input_filename, output_filename))


sess = tf.InteractiveSession()
convert_tfrecords("/path/to/libsvm/file/train.libsvm", "/path/to/tfrecord/file/train.tfrecords")
sess.close()
