# Import all needed libraries
import tensorflow as tf
import tfRecord_rw.tfrecord_reader as tfreader
import os


# Import all needed global variables
import tfRecord_rw.global_variables as global_var


def global_file_path_reader(parent_path,
                            ext):
    """
    Get the global path of all files under the directory parent_path that have the extension ext
    :param path: The global path of parent directory
    :param ext: The extension of wanted files under parent directory
    :return: A list contains all files paths under parent_path with extension ext
    """
    return [os.path.join(parent_path, f) for f in os.listdir(parent_path) if f.endswith(ext)]


def transfer_list_to_tfqueue(list):
    """
    Convert a list to tensorflow queue
    :param list: A list that is needed to convert
    :return: A converted tensorflow queue
    """
    queue = tf.RandomShuffleQueue(capacity=len(list), min_after_dequeue=0, dtypes=tf.string)
    queue.enqueue_many(tf.convert_to_tensor(list, dtype=tf.string))
    return queue


if __name__ == '__main__':
    # with tf.Session() as sess:
    #     list = global_file_path_reader(global_var.testPath, "tfrecord")
    #     queue = transfer_list_to_tfqueue(list)
    #     Atfreader = tfreader.YT8MFrameFeatureReader()
    #     features = Atfreader.prepare_reader(filename_queue=queue)
    #     print(sess.run(tf.report_uninitialized_variables()))
    #     sess.run(features)
    # print(queue.dequeue())

    with tf.Session() as sess:
        # Create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer(["/Users/leo/Academic/Youtube8M/test_path/train0a.tfrecord"],
                                                        num_epochs=1)
        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        contexts, feature = tf.parse_single_sequence_example(
            serialized_example,
            context_features={"video_id": tf.FixedLenFeature(
                [], tf.string),
                "labels": tf.VarLenFeature(tf.int64)},
            sequence_features={
                feature_name: tf.FixedLenSequenceFeature([], dtype=tf.string)
                for feature_name in ["inc3"]
            })

        # Decode the record read by the reader
        features = tf.parse_single_sequence_example(serialized_example, context_features=contexts, sequence_features=feature)
        # Convert the image data from string back to the numbers
        image = tf.decode_raw(features['video_id'], tf.float32)

        # Cast label data into int32
        label = tf.cast(features['labels'], tf.int32)
        # Reshape image data into the original shape
        image = tf.reshape(image, [224, 224, 3])

        # Any preprocessing here ...

        # Creates batches by randomly shuffling tensors
        images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1,
                                                min_after_dequeue=10)