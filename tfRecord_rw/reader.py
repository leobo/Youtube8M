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
    :return: A converted tensorflow FIFO queue
    """

    return tf.train.string_input_producer(list, capacity=1)


if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        file_name_list = global_file_path_reader("/Users/leo/Academic/Youtube8M/test_path/", "tfrecord")
        file_name_queue = transfer_list_to_tfqueue(file_name_list)
        batch_video_ids, batch_video_matrix, batch_labels, batch_frames = \
            tfreader.YT8MFrameFeatureReader().prepare_reader(filename_queue=file_name_queue)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run([batch_video_ids, batch_video_matrix, batch_labels, batch_frames])
        coord.request_stop()
        coord.join(threads)
        print(batch_frames)


