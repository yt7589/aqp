from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow_datasets as tfds
import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt

from ann.transformer.transformer_engine import TransformerEngine

class TransformerApp(object):
    BUFFER_SIZE = 20000
    BATCH_SIZE = 64
    MAX_LENGTH = 40

    def __init__(self):
        self.name = 'TransformEngine'

    def startup(self):
        train_dataset, val_dataset = self.load_dataset()
        transformer_engine = TransformerEngine()
        transformer_engine.train(
            train_dataset, val_dataset,
            self.tokenizer_en, self.tokenizer_pt
        )
        
    def load_dataset(self):
        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
        train_examples, val_examples = examples['train'], examples['validation']
        # tokenize
        self.tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in train_examples), target_vocab_size=2**13)
        self.tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in train_examples), target_vocab_size=2**13)
        sample_string = 'Transformer is awesome.'
        tokenized_string = self.tokenizer_en.encode(sample_string)
        print ('Tokenized string is {}'.format(tokenized_string))
        original_string = self.tokenizer_en.decode(tokenized_string)
        print ('The original string: {}'.format(original_string))
        assert original_string == sample_string
        #
        for ts in tokenized_string:
            print ('{} ----> {}'.format(ts, self.tokenizer_en.decode([ts])))
        train_dataset = train_examples.map(self.tf_encode)
        train_dataset = train_dataset.filter(self.filter_max_length)
        # cache the dataset to memory to get a speedup while reading from it.
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.shuffle(TransformerEngine.BUFFER_SIZE).padded_batch(
            TransformerEngine.BATCH_SIZE, padded_shapes=([-1], [-1]))
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


        val_dataset = val_examples.map(self.tf_encode)
        val_dataset = val_dataset.filter(self.filter_max_length).padded_batch(
            TransformerEngine.BATCH_SIZE, padded_shapes=([-1], [-1]))
        
        return train_dataset, val_dataset

    
    def encode(self, lang1, lang2):
        '''
        在输入和输出上加入开始和结束特殊字符
        '''
        lang1 = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            lang1.numpy()) + [self.tokenizer_pt.vocab_size+1]
        lang2 = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            lang2.numpy()) + [self.tokenizer_en.vocab_size+1]
        return lang1, lang2

    def filter_max_length(self, x, y, max_length=MAX_LENGTH):
        '''
        限制输入信号长度
        '''
        return tf.logical_and(tf.size(x) <= max_length,
                                tf.size(y) <= max_length)

    def tf_encode(self, pt, en):
        return tf.py_function(self.encode, [pt, en], [tf.int64, tf.int64])




