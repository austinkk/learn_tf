# for description of using tf.data

import tensorflow as tf

# 创建dataset对象读取数据
tf.data.Dataset.from_tensors()
tf.data.Dataset.from_tensor_slices()
tf.data.TFRecordDataset # 如果是tfrecord

# 用tf.data.Dataset对象的各种方法对其处理
Dataset.map()
Dataset.batch()

# 使用迭代器对象
Dataset.make_one_shot_iterator()
Iterator.initializer # 通过此指令初始化迭代器状态
Iterator.get_next() # 此指令返回下一个元素的tf.Tensor

# 一个dataset 有多个元素(样本),一个元素有多个组件(特征)
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
print (dataset1.output_types) # tf.float32
print (dataset1.output_shapes) # (10,)

dataset2 = tf.data.Dataset.from_tensor_slices(
            (tf.random_uniform([4]), tf.random_uniform([4, 100], maxval = 100, dtype = tf.int32))
        )
print (dataset2.output_types) # (tf.float32, tf.int32)
print (dataset2.output_shapes) # ((),(100,))

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print (dataset3.output_types) # (tf.float32, (tf.float32, tf.int32))
print (dataset3.output_shapes) # (10, ((),(100,)))

dataset = tf.data.Dataset.from_tensor_slices(
               {
               "a": tf.random_uniform([4]),
               "b": tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)
               }
           )
# print(dataset.output_types)  # ==> "{'a': tf.float32, 'b': tf.int32}"
# print(dataset.output_shapes)  # ==> "{'a': (), 'b': (100,)}"

dataset1.map(lambda x: ...)
dataset2.flat_map(lambda x, y: ...)
dataset3.filter(lambda x, (y, z): ...)

# 迭代器的作用
# 目前支持以下几种迭代器
# 1. 单次迭代器 3. 可重新初始化迭代器 4. 可feeding迭代器
dataset = tf.data.Dataset.range(100)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

for i in range(100):
    value = sess.run(next_element)
    assert i == value

# 2. 可初始化迭代器
max_value = tf.placeholder(tf.int64, shape = [])
dataset = tf.data.Dataset.range(max_value)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

sess.run(iterator.initializer, feed_dict = {max_value: 10})
for i in range(10):
    value = sess.run(next_element)
    assert i == value

# 3. 可重新初始化迭代器, 一个迭代器公用
training_dataset = tf.data.Dataset.range(100).map(
        lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
validation_dataset = tf.data.Dataset.range(50)

iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                           training_dataset.output_shapes)
next_element = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

for _ in range(20):
    sess.run(training_init_op)
    for _ in range(100):
        sess.run(next_element)

    sess.run(validation_init_op)
    for _ in range(50):
        sess.run(next_element)

# 可feeding 迭代器, 两个迭代器分开用
training_dataset = tf.data.Dataset.range(100).map(
        lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
validation_dataset = tf.data.Dataset.range(50)

handle = tf.placeholder(tf.string, shape = [])
iterator = tf.data.Iterator.from_string_handle(handle, training_dataset.output_types, training_dataset.output_shapes)
next_element = iterator.get_next()

training_iterator = training_dataset.make_one_shot_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()

training_handle = sess.run(training_iterator.string_handle())
validation_handle = sess.run(validation_iterator.string_handle())

while True:
    for _ in range(200):
        sess.run(next_element, feed_dict={
            handle: training_handle})
            })
    sess.run(validation_iterator.initializer)
    for _ in range(50):
        sess.run(next_element, feed_dict={
            handle: validation_handle})
            })

# 从iterator里读取数据
dataset = tf.data.Dataset.range(5)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
result = tf.add(next_element, next_element)

sess = tf.Session()
sess.run(iterator.initializer)
for _ in range(5):
    sess.run(result)
try:
    sess.run(result)
except tf.errors.OutOfRangeError:
    print ("End of dataset")

sess.run(iterator.initializer)
while True:
    try:
        sess.run(result)
    except: tf.errors.OutOfRangeError:
        break

# 基于NumPy数组构建Dataset
with np.load("/var/data/training_data.npy") as data:
      features = data["features"]
        labels = data["labels"]
        
        assert features.shape[0] == labels.shape[0]
        
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))

# 节约内存的方式
with np.load("/var/data/training_data.npy") as data:
      features = data["features"]
        labels = data["labels"]
        
        # Assume that each row of `features` corresponds to the same row as `labels`.
        assert features.shape[0] == labels.shape[0]
        
        features_placeholder = tf.placeholder(features.dtype, features.shape)
        labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
        
        dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
        # [Other transformations on `dataset`...]
        dataset = ...
        iterator = dataset.make_initializable_iterator()
        
        sess.run(iterator.initializer, feed_dict={
            features_placeholder: features,
            labels_placeholder: labels})

# 基于tf.data.TFRecordDataset 构建 Dataset
filenames = ['a.tfrecord', 'b.tfrecord']
dataset = tf.data.TFRecordDataset(filenames) # 参数可以是字符串，字符串列表，字符串tf.tensor
# 使用tf.placeholder(tf.string)


filenames = tf.placeholder(tf.string, shape = [None])
dataset = tf.data.TFRecordDataset(filenames)

dataset = dataset.map(...)
dataset = dataset.repeat()
dataset = dataset.batch(32)
iterator = dataset.make_initializable_iterator()

training_filenames = []
sess.run(iterator.initializer, feed_dict={
    filenames: training_filenames})
    })
validation_filenames = []
sess.run(iterator.initializer, feed_dict={
    filenames: validation_filenames})
    })

# 读二进制文件
filenames = ["/var/data/file1.bin", "/var/data/file2.bin"]
dataset = tf.data.FixedLengthRecordDataset(filenames, record_bytes, header_bytes, footer_bytes， buffer_size)
"""
filenames ： tf.string，包含一个或多个文件名;
record_bytes ：tf.int64，一个 record 占的 bytes;
header_bytes ：（可选）tf.int64，每个文件开头需要跳过多少 bytes;
footer_bytes ：（可选）tf.int64，每个文件结尾需要忽略多少 bytes;
buffer_size ：（可选）tf.int64，读取时，缓冲多少bytes;
"""

# 从tf.Example中解析数据
def _parse_function(example_proto):
    features = {
                "image": tf.FixedLenFeature((), tf.string, default_value=""),
                "label": tf.FixedLenFeature((), tf.int32, default_value=0)
               }
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features["image"], parsed_features["label"]
                  
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_function)

# 解码图片数据并调整其大小 / 直接从文件读取文件
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string)
    image_resized = tf.image.resize_images(image_decoded, [28, 28])
    return image_resized, label

filenames = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg", ...])
labels = tf.constant([0, 37, ...])
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_function)

# tf.py_func
tf.py_func(func, # 一个Python函数
           inp, # 一个Tensor列表
           Tout, # 输出的Tensor的dtype或Tensors的dtype列表
           stateful=True, # 布尔值，输入值相同，输出值就相同，那么就将stateful设置为False
           name=None)

import cv2

# Use a custom OpenCV function to read the image, instead of the standard
# TensorFlow `tf.read_file()` operation.
def _read_py_function(filename, label):
  image_decoded = cv2.imread(filename.decode(), cv2.IMREAD_GRAYSCALE)
  return image_decoded, label

# Use standard TensorFlow operations to resize the image to a fixed shape.
def _resize_function(image_decoded, label):
  image_decoded.set_shape([None, None, None])
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

filenames = ["/var/data/image1.jpg", "/var/data/image2.jpg", ...]
labels = [0, 37, 29, 1, ...]

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(
    lambda filename, label: tuple(tf.py_func(
        _read_py_function, [filename, label], [tf.uint8, label.dtype])))
dataset = dataset.map(_resize_function)

# 数据集进行batch
inc_dataset = tf.data.Dataset.range(100)
dec_dataset = tf.data.Dataset.range(0, -100, -1)
dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
batched_dataset = dataset.batch(4)

iterator = batched_dataset.make_one_shot_iterator()
next_element = iterator.get_next()

print(sess.run(next_element))  # ==> ([0, 1, 2,   3],   [ 0, -1,  -2,  -3])
print(sess.run(next_element))  # ==> ([4, 5, 6,   7],   [-4, -5,  -6,  -7])
print(sess.run(next_element))  # ==> ([8, 9, 10, 11],   [-8, -9, -10, -11])

dataset = tf.data.Dataset.range(100)
dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
dataset = dataset.padded_batch(4, padded_shapes=[None])

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

print(sess.run(next_element))  # ==> [[0, 0, 0], 
                               #      [1, 0, 0], 
                               #      [2, 2, 0], 
                               #      [3, 3, 3]]

print(sess.run(next_element))  # ==> [[4, 4, 4, 4, 0, 0, 0],
                               #      [5, 5, 5, 5, 5, 0, 0],
                               #      [6, 6, 6, 6, 6, 6, 0],
                               #      [7, 7, 7, 7, 7, 7, 7]]

# 
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(...)
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(32)
dataset = dataset.repeat()
