#coding: utf-8
"""
make tfrecord more easy to use.
"""
import tensorflow as tf
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class TFRecordWriter:
    def __init__(self, record_path, log_num = 10000):
        self.writer = tf.python_io.TFRecordWriter(record_path)
        self.feature = {}
        self.counter = 0
        self.record_path = record_path
        self.log_num = log_num

    # 支持3个类型, byte_list/float_list/int64_list
    def int64_feature(self, value):
        return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))
    
    def bytes_feature(self, value):
        return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))
    
    def float_feature(self, value):
        return tf.train.Feature(float_list = tf.train.FloatList(value = [value]))

    def close(self):
        print ('%s has write %d records' % (self.record_path, self.counter))
        self.writer.close()

    def write(self):
        """
        example 协议块
        message Example {
            Features features = 1;
        };
        
        message Features {
            map<string, Feature> feature = 1;
        };
        
        message Feature {
            oneof kind {
                BytesList byte_list = 1;
                FloatList float_list = 2;
                Int64List int64_list = 3;
            }
        }
        """
        tf_example = tf.train.Example(
            features = tf.train.Features(
                feature = self.feature
            )
        )
        self.counter += 1
        if self.counter % self.log_num == 0:
            print ('%s has write %d records' % (self.record_path, self.counter))
        self.writer.write(tf_example.SerializeToString())

    def update(self, name, value, datatype):
        # datatype in ['byte', 'int', 'float']
        if datatype == 'byte':
            self.feature[name] = self.bytes_feature(value)
        elif datatype == 'int':
            self.feature[name] = self.int64_feature(value)
        elif datatype == 'float':
            self.feature[name] = self.float_feature(value)
        else:
            raise Exception("type error")



class TFRecordReader:
    def __init__(self, filenames):
        """
        使用local_variables_initializer()初始化局部变量
        tf.train.string_input_producer(
            string_tensor,
            num_epochs=None,
            shuffle=True,
            seed=None,
            capacity=32,
            shared_name=None,
            name=None,
            cancel_op=None
        )
        """
        self.dataset = tf.data.TFRecordDataset(filenames)

    @staticmethod
    def parse_one_example(serialized_example):
        """
        每次使用需要修改
        """
        # image_batch, label_batch = tf.train.batch([image_tensor, label], batch_size=10, num_threads=1, capacity=10)
        # 返回字典
        example = tf.parse_single_example(serialized_example,
                                           features = {
                                                   "userid" : tf.FixedLenFeature((), tf.int64, default_value = 0),
                                                   "height" : tf.FixedLenFeature((), tf.float32),
                                                   "label"  : tf.FixedLenFeature((), tf.string)
                                               }
                                          )
        return (example["userid"], example["height"]), example["label"]

    def parse(self):
        self.dataset = self.dataset.map(self.__class__.parse_one_example)

    def shuffle(self, buffer_size = 100):
        self.dataset = self.dataset.shuffle(buffer_size = buffer_size)

    def batch(self, batch_size = 32):
        self.dataset = self.dataset.batch(batch_size)

    def repeat(self, repeat_times = None):
        if repeat_times == None:
            self.dataset = self.dataset.repeat()
        else:
            self.dataset = self.dataset.repeat(repeat_times)

    def get_next_element(self):
        iterator = self.dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        return next_element

if __name__ == '__main__':
    test_data = [
                [1, 1.2, b'taxi'],
                [2, 1.3, b'car'],
                [3, 1.4, b'car'],
                [4, 1.5, b'taxi']
            ]
    tfwriter = TFRecordWriter('./test.tfrecord')
    for line in test_data:
        tfwriter.update('userid', line[0], 'int')
        tfwriter.update('height', line[1], 'float')
        tfwriter.update('label', line[2], 'byte')
        tfwriter.write()
    tfwriter.close()

    tfreader = TFRecordReader(['./test.tfrecord'])

    tfreader.parse()
    tfreader.batch(2)
    tfreader.repeat(2)
    print (tfreader.dataset)
    next_element = tfreader.get_next_element()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        try:
            while True:
                x = sess.run(next_element)
                print (x)
        except tf.errors.OutOfRangeError:
            print ('Done training -- epoch limit reached')

