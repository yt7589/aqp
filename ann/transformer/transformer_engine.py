#
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow_datasets as tfds
import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt

from ann.transformer.transformer_util import TransformerUtil
from ann.transformer.multi_head_attention import MultiHeadAttention
from ann.transformer.encoder_layer import EncoderLayer
from ann.transformer.encoder import Encoder
from ann.transformer.decoder_layer import DecoderLayer
from ann.transformer.decoder import Decoder
from ann.transformer.transformer import Transformer
from ann.transformer.custom_schedule import CustomSchedule

#@tf.function
def train_step(transformer, loss_object,
            optimizer, train_loss, train_accuracy,
            inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    
    enc_padding_mask, combined_mask, dec_padding_mask = \
                TransformerUtil.create_masks(inp, tar_inp)
    
    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp, 
                                    True, 
                                    enc_padding_mask, 
                                    combined_mask, 
                                    dec_padding_mask)
        loss = TransformerUtil.loss_function(
                        loss_object, tar_real, predictions)
    gradients = tape.gradient(loss, transformer.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    train_loss(loss)
    train_accuracy(tar_real, predictions)

class TransformerEngine(object):
    BUFFER_SIZE = 20000
    BATCH_SIZE = 64
    MAX_LENGTH = 40

    def __init__(self):
        self.name = 'Transformer'
        self.checkpoint_path = "./work/transformer_enpt"
        self.ckpt_manager = None

    def train(self, train_dataset, val_dataset, tokenizer_en, tokenizer_pt):
        transformer, train_loss, train_accuracy, loss_object, optimizer = self.build_model(train_dataset, val_dataset, tokenizer_en, tokenizer_pt)
        EPOCHS = 2 #20
        for epoch in range(EPOCHS):
            start = time.time()
            train_loss.reset_states()
            train_accuracy.reset_states()
            # inp -> portuguese, tar -> english
            print('epoch:{0}'.format(epoch))
            for (batch, (inp, tar)) in enumerate(train_dataset):
                train_step(transformer, loss_object,
                        optimizer, train_loss, train_accuracy,
                        inp, tar
                )
                print('      batch:{0}'.format(batch))
                if batch % 500 == 0:
                    print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch + 1, batch, train_loss.result(), 
                        train_accuracy.result())
                    )
            if epoch > 0: #if (epoch + 1) % 5 == 0 or epoch > 0:
                ckpt_save_path = self.ckpt_manager.save()
                print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                    ckpt_save_path))
            print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                            train_loss.result(), 
                                                            train_accuracy.result()))
            print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

        print('Training finished ^_^')
 
    def run(self, transformer, sentence):
        return self.translate(transformer, sentence)



    def build_model(self, train_dataset, val_dataset, tokenizer_en, tokenizer_pt):
        self.tokenizer_en = tokenizer_en
        self.tokenizer_pt = tokenizer_pt
        pt_batch, en_batch = next(iter(val_dataset))

        num_layers = 4
        d_model = 128
        dff = 512
        num_heads = 8

        input_vocab_size = tokenizer_pt.vocab_size + 2
        target_vocab_size = tokenizer_en.vocab_size + 2
        dropout_rate = 0.1

        learning_rate = CustomSchedule(d_model)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                            epsilon=1e-9)

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction='none')
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')
        transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size, dropout_rate)

        self.ckpt = tf.train.Checkpoint(transformer=transformer,
                                optimizer=optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=5)
        # if a checkpoint exists, restore the latest checkpoint.
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!')
        return transformer, train_loss, train_accuracy, loss_object, optimizer
    


    def evaluate(self, transformer, inp_sentence):
        start_token = [self.tokenizer_pt.vocab_size]
        end_token = [self.tokenizer_pt.vocab_size + 1]
        
        # inp sentence is portuguese, hence adding the start and end token
        inp_sentence = start_token + self.tokenizer_pt.encode(inp_sentence) + end_token
        encoder_input = tf.expand_dims(inp_sentence, 0)
        
        # as the target is english, the first word to the transformer should be the
        # english start token.
        decoder_input = [self.tokenizer_en.vocab_size]
        output = tf.expand_dims(decoder_input, 0)
            
        for i in range(TransformerEngine.MAX_LENGTH):
            enc_padding_mask, combined_mask, dec_padding_mask = TransformerUtil.create_masks(
                encoder_input, output)
        
            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = transformer(encoder_input, 
                                                        output,
                                                        False,
                                                        enc_padding_mask,
                                                        combined_mask,
                                                        dec_padding_mask)
            
            # select the last word from the seq_len dimension
            predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            
            # return the result if the predicted_id is equal to the end token
            if tf.equal(predicted_id, self.tokenizer_en.vocab_size+1):
                return tf.squeeze(output, axis=0), attention_weights
            
            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0), attention_weights

    def translate(self, transformer, sentence):
        result, attention_weights = self.evaluate(transformer, sentence)
        predicted_sentence = self.tokenizer_en.decode([i for i in result 
                                                    if i < self.tokenizer_en.vocab_size])
        return format(predicted_sentence), attention_weights, result
        #if plot:
        #   self.plot_attention_weights(attention_weights, sentence, result, plot)

    











    def study_model(self, train_dataset, val_dataset, tokenizer_en, tokenizer_pt):
        self.tokenizer_en = tokenizer_en
        self.tokenizer_pt = tokenizer_pt
        pt_batch, en_batch = next(iter(val_dataset))
        print('{0} vs {1}'.format(pt_batch, en_batch))

        pos_encoding = TransformerUtil.positional_encoding(50, 512)
        print (pos_encoding.shape)

        plt.pcolormesh(pos_encoding[0], cmap='RdBu')
        plt.xlabel('Depth')
        plt.xlim((0, 512))
        plt.ylabel('Position')
        plt.colorbar()
        plt.show()

        x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
        x001 = TransformerUtil.create_padding_mask(x)
        print(x001)
        x = tf.random.uniform((1, 3))
        temp = TransformerUtil.create_look_ahead_mask(x.shape[1])
        print(temp)
        print('######## 111111 ###############')

        np.set_printoptions(suppress=True)
        temp_k = tf.constant([[10,0,0],
                            [0,10,0],
                            [0,0,10],
                            [0,0,10]], dtype=tf.float32)  # (4, 3)
        temp_v = tf.constant([[   1,0],
                            [  10,0],
                            [ 100,5],
                            [1000,6]], dtype=tf.float32)  # (4, 2)
        # This `query` aligns with the second `key`,
        # so the second `value` is returned.
        temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
        self.print_out(temp_q, temp_k, temp_v)

        print('************ 22222222 *******************')
        temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)  # (1, 3)
        self.print_out(temp_q, temp_k, temp_v)

        print('############ 3333333 ######################')
        temp_q = tf.constant([[10, 10, 0]], dtype=tf.float32)  # (1, 3)
        self.print_out(temp_q, temp_k, temp_v)

        print('############# 444444444 ####################')
        temp_q = tf.constant([[0, 0, 10], [0, 10, 0], [10, 10, 0]], dtype=tf.float32)  # (3, 3)
        self.print_out(temp_q, temp_k, temp_v)

        print('############## 555555555 ################')
        temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
        y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
        out, attn = temp_mha(y, k=y, q=y, mask=None)
        print('{0}   {1}'.format(out.shape, attn.shape))

        print('########### 66666666 #################')
        sample_ffn = TransformerUtil.point_wise_feed_forward_network(512, 2048)
        print(sample_ffn(tf.random.uniform((64, 50, 512))).shape)

        print('############ 7777777777 ###############')
        sample_encoder_layer = EncoderLayer(512, 8, 2048)
        sample_encoder_layer_output = sample_encoder_layer(
            tf.random.uniform((64, 43, 512)), False, None)
        print(sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)

        print('########## 8888888888 ##################')
        sample_decoder_layer = DecoderLayer(512, 8, 2048)
        sample_decoder_layer_output, _, _ = sample_decoder_layer(
            tf.random.uniform((64, 50, 512)), sample_encoder_layer_output, 
            False, None, None)
        print(sample_decoder_layer_output.shape)  # (batch_size, target_seq_len, d_model)

        print('######### 999 #########')
        sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8, 
                         dff=2048, input_vocab_size=8500)
        sample_encoder_output = sample_encoder(tf.random.uniform((64, 62)), 
                                            training=False, mask=None)
        print (sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)

        print('######### 10 10 10 #######')
        sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8, 
                         dff=2048, target_vocab_size=8000)
        output, attn = sample_decoder(tf.random.uniform((64, 26)), 
                                    enc_output=sample_encoder_output, 
                                    training=False, look_ahead_mask=None, 
                                    padding_mask=None)
        print('{0};  {1}'.format(output.shape, attn['decoder_layer2_block2'].shape))

        print('######### 11 11 11 ############')
        sample_transformer = Transformer(
                num_layers=2, d_model=512, num_heads=8, dff=2048, 
                input_vocab_size=8500, target_vocab_size=8000)
        temp_input = tf.random.uniform((64, 62))
        temp_target = tf.random.uniform((64, 26))
        fn_out, _ = sample_transformer(temp_input, temp_target, training=False, 
                                    enc_padding_mask=None, 
                                    look_ahead_mask=None,
                                    dec_padding_mask=None)
        print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)

        num_layers = 4
        d_model = 128
        dff = 512
        num_heads = 8

        input_vocab_size = tokenizer_pt.vocab_size + 2
        target_vocab_size = tokenizer_en.vocab_size + 2
        dropout_rate = 0.1

        learning_rate = CustomSchedule(d_model)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                            epsilon=1e-9)
        temp_learning_rate_schedule = CustomSchedule(d_model)

        plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
        plt.ylabel("Learning Rate")
        plt.xlabel("Train Step")
        plt.show()

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction='none')
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')
        transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size, dropout_rate)

        checkpoint_path = "./work/transformer_enpt"
        ckpt = tf.train.Checkpoint(transformer=transformer,
                                optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!')

        EPOCHS = 2 #20

        for epoch in range(EPOCHS):
            start = time.time()
            train_loss.reset_states()
            train_accuracy.reset_states()
            # inp -> portuguese, tar -> english
            print('epoch:{0}'.format(epoch))
            for (batch, (inp, tar)) in enumerate(train_dataset):
                train_step(transformer, loss_object,
                        optimizer, train_loss, train_accuracy,
                        inp, tar
                )
                print('      batch:{0}'.format(batch))
                if batch % 500 == 0:
                    print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch + 1, batch, train_loss.result(), 
                        train_accuracy.result())
                    )
            if epoch > 0: #if (epoch + 1) % 5 == 0 or epoch > 0:
                ckpt_save_path = ckpt_manager.save()
                print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                    ckpt_save_path))
            print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                            train_loss.result(), 
                                                            train_accuracy.result()))
            print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

        self.translate(transformer, "este é um problema que temos que resolver.")
        print ("Real translation: this is a problem we have to solve .")
        self.translate(transformer, "os meus vizinhos ouviram sobre esta ideia.")
        print ("Real translation: and my neighboring homes heard about this idea .")
        self.translate(transformer, "vou então muito rapidamente partilhar convosco algumas histórias de algumas coisas mágicas que aconteceram.")
        print ("Real translation: so i 'll just share with you some stories very quickly of some magical things that have happened .")
        self.translate(transformer, "este é o primeiro livro que eu fiz.", plot='decoder_layer4_block2')
        print ("Real translation: this is the first book i've ever done.")
























    def plot_attention_weights(self, attention, sentence, result, layer):
        fig = plt.figure(figsize=(16, 8))
        sentence = self.tokenizer_pt.encode(sentence)
        attention = tf.squeeze(attention[layer], axis=0)
        for head in range(attention.shape[0]):
            ax = fig.add_subplot(2, 4, head+1)
            # plot the attention weights
            ax.matshow(attention[head][:-1, :], cmap='viridis')
            fontdict = {'fontsize': 10}
            ax.set_xticks(range(len(sentence)+2))
            ax.set_yticks(range(len(result)))
            ax.set_ylim(len(result)-1.5, -0.5)
            ax.set_xticklabels(
                ['<start>']+[self.tokenizer_pt.decode([i]) for i in sentence]+['<end>'], 
                fontdict=fontdict, rotation=90)
            ax.set_yticklabels([self.tokenizer_en.decode([i]) for i in result 
                                if i < self.tokenizer_en.vocab_size], 
                            fontdict=fontdict)
            ax.set_xlabel('Head {}'.format(head+1))
        plt.tight_layout()
        plt.show()


    def print_out(self, q, k, v):
        temp_out, temp_attn = TransformerUtil.scaled_dot_product_attention(
            q, k, v, None)
        print ('Attention weights are:')
        print (temp_attn)
        print ('Output is:')
        print (temp_out)

    