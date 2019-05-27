#
import numpy as np
import tensorflow as tf

class NlpTfrecordDataset(object):
    feature_description = {
            'en': tf.io.VarLenFeature(tf.string), 
            'pt': tf.io.VarLenFeature(tf.string),
        }

    def __init__(self):
        self.name = 'NlpTfrecordDataset'

    def load_pt_en_ds(self, ds_files):
        raw_dataset = tf.data.TFRecordDataset(ds_files)
        # Create a description of the features.
        idx = 1
        parsed_dataset = raw_dataset.map(self._parse_function)
        pts_l = []
        ens_l = []
        for item in parsed_dataset:
            print('{0} => {1}\r\n'.format(
                str(item['pt'].values[0].numpy(), encoding='utf-8'),
                str(item['en'].values[0].numpy(), encoding='utf-8')
            ))
            pts_l.append(str(item['pt'].values[0].numpy(), encoding='utf-8'))
            ens_l.append(str(item['en'].values[0].numpy(), encoding='utf-8'))
            idx += 1
            if idx > 5:
                break

        pts = np.array(pts_l)
        ens = np.array(ens_l)
        self.features_dataset = tf.data.Dataset.from_tensor_slices((pts, ens))
        serialized_features_dataset = self.features_dataset.map(self.tf_serialize_example)
        filename = './work/test1.tfrecord'
        writer = tf.data.experimental.TFRecordWriter(filename)
        writer.write(serialized_features_dataset)

    def load(self, ds_files):
        raw_dataset = tf.data.TFRecordDataset(ds_files)
        print(raw_dataset)
        idx = 0
        # Create a description of the features.
        parsed_dataset = raw_dataset.map(self._parse_function)
        print(parsed_dataset)
        for item in parsed_dataset:
            print(repr(item))
            print(str(item['pt'].numpy(), encoding='utf-8'))
            idx += 1
            if idx > 5:
                break

    def write_to_file(self, ds_file):
        print('英文属性：{0}'.format(self._bytes_feature(b'Very good!')))
        feature = self._bytes_feature(u'非常好的事情！'.encode('utf-8'))
        print('中文属性：{0}'.format(feature))
        print('中文属性序列化：{0}'.format(feature.SerializeToString()))
        print('浮点数属性：{0}'.format(self._float_feature(np.exp(1))))
        print('整数属性：{0}'.format(self._int64_feature(True)))
        print('整数属性序列化：{0}'.format(self._int64_feature(1).SerializeToString()))

        pt1 = '人工智能1'
        en1 = 'Artificial Intelligence 1'
        serialized_example = self.serialize_example(pt1.encode('utf-8'), en1.encode('utf-8'))
        print('序列化pt1、en1：{0}'.format(serialized_example))
        example_proto = tf.train.Example.FromString(serialized_example)
        print('反序列化pt1、en1：{0}'.format(example_proto))

        pts = np.array(['人1', '人2', '人3'])
        ens = np.array(['p1', 'p2', 'p3'])
        self.features_dataset = tf.data.Dataset.from_tensor_slices((pts, ens))
        print('真实数据集(np.array)：{0}'.format(self.features_dataset))
        print('遍历tfrecords：')
        for pt1, en1 in self.features_dataset:
            print('{0} => {1}'.format(str(pt1.numpy(), encoding='utf-8'), en1))
        
        serialized_features_dataset = self.features_dataset.map(self.tf_serialize_example)
        print('serialized_features_dataset:{0}'.format(serialized_features_dataset))
        filename = './work/test.tfrecord'
        writer = tf.data.experimental.TFRecordWriter(filename)
        writer.write(serialized_features_dataset)

        

    
    def _parse_function(self, example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, 
                    NlpTfrecordDataset.feature_description)

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def serialize_example(self, pt, en):
        """
        Creates a tf.Example message ready to be written to a file.
        """
        # Create a dictionary mapping the feature name to the tf.Example-compatible
        # data type.
        feature = {
            'pt': self._bytes_feature(pt),
            'en': self._bytes_feature(en),
        }

        # Create a Features message using tf.train.Example.

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def tf_serialize_example(self, pt, en):
        tf_string = tf.py_function(
            self.serialize_example,
            (pt, en),  # pass these args to the above function.
            tf.string
        )      # the return type is `tf.string`.
        return tf.reshape(tf_string, ()) # The result is a scalar

    def generator(self):
        for features in self.features_dataset:
            yield self.serialize_example(*features)