#
import os
import tensorflow as tf
from app.nlp.transformer_app import TransformerApp
from app.nlp.nlp_tfrecord_dataset import NlpTfrecordDataset

class NlpMain(object):
    def __init__(self):
        self.name = 'NlpMain'

    def startup(self):
        print('自然语言处理Transformer模型')
        #te = TransformerApp()
        #te.startup()

        ds_files = ['/Users/arxanfintech/tensorflow_datasets/ted_hrlr_translate/pt_to_en/0.0.1/ted_hrlr_translate-train.tfrecord-00000-of-00001']
        ds = NlpTfrecordDataset()
        ds.load_pt_en_ds(ds_files)
        #ds.write_to_file('')
