from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow_datasets as tfds
import tensorflow as tf
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

from ann.transformer.transformer_engine import TransformerEngine
from ann.transformer.transformer_util import TransformerUtil
from ann.transformer.multi_head_attention import MultiHeadAttention
from ann.transformer.custom_schedule import CustomSchedule

class TransformerApp(object):
    MODE_TRAIN = 1
    MODE_RUN = 2
    BUFFER_SIZE = 20000
    BATCH_SIZE = 64
    MAX_LENGTH = 40

    def __init__(self):
        self.name = 'TransformEngine'

    def t1(self):
        train_dataset, val_dataset = self.load_dataset()
        start_token = [self.tokenizer_pt.vocab_size]
        end_token = [self.tokenizer_pt.vocab_size+1]
        inp_sentence = 'este é um problema que temos que resolver.'
        inp_sentence = start_token + self.tokenizer_pt.encode(inp_sentence) + end_token
        print('\r\n*************\r\n****************\r\n****************')
        print('inp_sentence type:{0}----{1}'.format(type(inp_sentence), inp_sentence))
        encoder_input = tf.expand_dims(inp_sentence, 0)
        print('encoder_input type:{0}\r\n{1}'.format(type(encoder_input), encoder_input))
        decoder_input = [self.tokenizer_en.vocab_size]
        print('decoder_input:{0} {1}'.format(type(decoder_input), decoder_input))
        output = tf.expand_dims(decoder_input, 0)
        print('output:{0} {1}'.format(type(output), output))
        print('i=0')

        transformer_engine = TransformerEngine()
        transformer, train_loss, train_accuracy, loss_object, optimizer = \
                        transformer_engine.build_model(
                            train_dataset, val_dataset, 
                            self.tokenizer_en, self.tokenizer_pt
                        )

        i = 0
        enc_padding_mask, combined_mask, dec_padding_mask = TransformerUtil.create_masks(
                encoder_input, output)
        print('enc_padding_mask:{0}; {1}'.format(enc_padding_mask.shape, enc_padding_mask))
        print('combined_mask:{0}; {1}'.format(combined_mask.shape, combined_mask))
        print('dec_padding_mask:{0}; {1}'.format(dec_padding_mask.shape, dec_padding_mask))
        predictions, attention_weights = transformer(encoder_input, 
                                                        output,
                                                        False,
                                                        enc_padding_mask,
                                                        combined_mask,
                                                        dec_padding_mask)
        print('predictions:{0}; {1}'.format(predictions.shape, predictions))
        predictions = predictions[: ,-1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        output = tf.concat([output, predicted_id], axis=-1)
        print('id:{0}, output:{1}; {2}'.format(predicted_id, output.shape, output))
        print('\r\ni=1\r\n')
        # 
        i = 1
        enc_padding_mask, combined_mask, dec_padding_mask = TransformerUtil.create_masks(
                encoder_input, output)
        print('enc_padding_mask:{0}; {1}'.format(enc_padding_mask.shape, enc_padding_mask))
        print('combined_mask:{0}; {1}'.format(combined_mask.shape, combined_mask))
        print('dec_padding_mask:{0}; {1}'.format(dec_padding_mask.shape, dec_padding_mask))
        predictions, attention_weights = transformer(encoder_input, 
                                                        output,
                                                        False,
                                                        enc_padding_mask,
                                                        combined_mask,
                                                        dec_padding_mask)
        print('predictions:{0}; {1}'.format(predictions.shape, predictions))
        predictions = predictions[: ,-1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        output = tf.concat([output, predicted_id], axis=-1)
        print('id:{0}, output:{1}; {2}'.format(predicted_id, output.shape, output))
        print('\r\ni=2\r\n')
        #
        i = 2
        enc_padding_mask, combined_mask, dec_padding_mask = TransformerUtil.create_masks(
                encoder_input, output)
        print('enc_padding_mask:{0}; {1}'.format(enc_padding_mask.shape, enc_padding_mask))
        print('combined_mask:{0}; {1}'.format(combined_mask.shape, combined_mask))
        print('dec_padding_mask:{0}; {1}'.format(dec_padding_mask.shape, dec_padding_mask))
        predictions, attention_weights = transformer(encoder_input, 
                                                        output,
                                                        False,
                                                        enc_padding_mask,
                                                        combined_mask,
                                                        dec_padding_mask)
        print('predictions:{0}; {1}'.format(predictions.shape, predictions))
        predictions = predictions[: ,-1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        output = tf.concat([output, predicted_id], axis=-1)
        print('id:{0}, output:{1}; {2}'.format(predicted_id, output.shape, output))
        print('\r\ni=3\r\n')
        i = 3
        enc_padding_mask, combined_mask, dec_padding_mask = TransformerUtil.create_masks(
                encoder_input, output)
        print('enc_padding_mask:{0}; {1}'.format(enc_padding_mask.shape, enc_padding_mask))
        print('combined_mask:{0}; {1}'.format(combined_mask.shape, combined_mask))
        print('dec_padding_mask:{0}; {1}'.format(dec_padding_mask.shape, dec_padding_mask))
        predictions, attention_weights = transformer(encoder_input, 
                                                        output,
                                                        False,
                                                        enc_padding_mask,
                                                        combined_mask,
                                                        dec_padding_mask)
        print('predictions:{0}; {1}'.format(predictions.shape, predictions))
        predictions = predictions[: ,-1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        output = tf.concat([output, predicted_id], axis=-1)
        print('id:{0}, output:{1}; {2}'.format(predicted_id, output.shape, output))

    def startup(self):
        self.t1()
        i_debug = 1
        if 1 == i_debug:
            return
        mode = TransformerApp.MODE_RUN
        train_dataset, val_dataset = self.load_dataset()
        transformer_engine = TransformerEngine()
        if TransformerApp.MODE_TRAIN == mode:
            transformer_engine.train(
                train_dataset, val_dataset,
                self.tokenizer_en, self.tokenizer_pt
            )
        else:
            transformer, train_loss, train_accuracy, loss_object, optimizer = \
                        transformer_engine.build_model(
                            train_dataset, val_dataset, 
                            self.tokenizer_en, self.tokenizer_pt
                        )
            # first
            print('pt: este é um problema que temos que resolver.')
            translated, _, _ = transformer_engine.run(transformer, 'este é um problema que temos que resolver.')
            print('translated: {0}'.format(translated))
            print('Real translation: this is a problem we have to solve .\r\n*****\r\n')
            # second
            translated, _, _ = transformer_engine.run(transformer, 'os meus vizinhos ouviram sobre esta ideia.')
            print('translated: {0}'.format(translated))
            print('Real translation: and my neighboring homes heard about this idea .\r\n*****\r\n')
            # third
            translated, _, _ = transformer_engine.run(transformer, 'vou então muito rapidamente partilhar convosco algumas histórias de algumas coisas mágicas que aconteceram.')
            print('translated: {0}'.format(translated))
            print('Real translation: so i \'ll just share with you some stories very quickly of some magical things that have happened .\r\n*****\r\n')
            # forth
            sentence = 'este é o primeiro livro que eu fiz.'
            translated, attention_weights, result = transformer_engine.run(transformer, sentence)
            print('translated: {0}'.format(translated))
            print('Real translation: this is the first book i\'ve ever done.\r\n*****\r\n')
            plot = 'decoder_layer4_block2'
            transformer_engine.plot_attention_weights(attention_weights, sentence, result, plot)
        
    def load_dataset(self):
        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
        train_examples, val_examples = examples['train'], examples['validation']
        print(train_examples)
        for item in train_examples.take(5):
            print('src:{0}'.format(item[0].numpy()))
            print('## dest:{0}'.format(item[1].numpy()))
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

    def test_scaled_dot_product_attention(self):
        # 3 * 5
        X = np.array([
            [1.1, 1.2, 1.3, 1.4, 1.5],
            [2.1, 2.2, 2.3, 2.4, 2.5],
            [3.1, 3.2, 3.3, 3.4, 3.5]
        ], dtype=np.float32)
        print('X:{0}!'.format(X.shape))
        W_Q = np.array([
            [11.1, 11.2, 11.3, 11.4],
            [12.1, 12.2, 12.3, 12.4],
            [13.1, 13.2, 13.3, 13.4],
            [14.1, 14.2, 14.3, 14.4],
            [15.1, 15.2, 15.3, 15.4]
        ], dtype=np.float32)
        print('W_Q:{0}!'.format(W_Q.shape))
        W_K = np.array([
            [21.1, 21.2, 21.3, 21.4],
            [22.1, 22.2, 22.3, 22.4],
            [23.1, 23.2, 23.3, 23.4],
            [24.1, 24.2, 24.3, 24.4],
            [25.1, 25.2, 25.3, 25.4]
        ], dtype=np.float32)
        print('W_K:{0}!'.format(W_K.shape))
        W_V = np.array([
            [31.1, 31.2, 31.3, 31.4],
            [32.1, 32.2, 32.3, 32.4],
            [33.1, 33.2, 33.3, 33.4],
            [34.1, 34.2, 34.3, 34.4],
            [35.1, 35.2, 35.3, 35.4]
        ], dtype=np.float32)
        print('W_V:{0}!'.format(W_V.shape))
        Q = np.matmul(X, W_Q)
        print('Q:{0}!'.format(Q.shape))
        K = np.matmul(X, W_K)
        print('K:{0}!'.format(K.shape))
        V = np.matmul(X, W_V)
        print('V:{0}!'.format(V.shape))
        Z, attn = TransformerUtil.scaled_dot_product_attention(
            Q, K, V, None)
        print('Z:{0}!'.format(Z.shape))
        print('attn:{0}!'.format(attn.shape))

    def test_multi_head_attention(self):
        temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
        y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
        out, attn = temp_mha(y, k=y, q=y, mask=None)
        print('{0}   {1}'.format(out.shape, attn.shape))

    def test_custom_schedule(self):
        d_model = 128
        temp_learning_rate_schedule = CustomSchedule(d_model)
        plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
        plt.ylabel("Learning Rate")
        plt.xlabel("Train Step")
        plt.show()


