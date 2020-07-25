"""Splitting TFR dataset in train, test and val"""

import tensorflow as tf
tf.enable_eager_execution()

class SplittingData():

    def __init__(self, input_file, percentaje_train, percentaje_test, percentaje_val, \
                 file_out_tfr_train, file_out_tfr_test, file_out_tfr_val):
        self.input_file = input_file
        self.percentaje_train = percentaje_train
        self.percentaje_test = percentaje_test
        self.percentaje_val = percentaje_val
        self.file_out_tfr_train = file_out_tfr_train
        self.file_out_tfr_test = file_out_tfr_test
        self.file_out_tfr_val = file_out_tfr_val

    def save_tfr_dataset(self, dataset, file_out_tfr):

        with tf.io.TFRecordWriter(file_out_tfr) as writer:
            for record in dataset:
                writer.write(record.numpy())


    def run(self):

        full_dataset = tf.data.TFRecordDataset(self.input_file)
        full_dataset_size = sum(1 for record in full_dataset)
        print("records_full_dataset = {}".format(full_dataset_size))
        train_size = int(self.percentaje_train * full_dataset_size)
        val_size = int(self.percentaje_val * full_dataset_size)
        test_size = int(self.percentaje_test * full_dataset_size)

        full_dataset = full_dataset.shuffle(128)
        train_dataset = full_dataset.take(train_size)
        train_dataset_size = sum(1 for record in train_dataset)
        print("train_dataset_size = {}".format(train_dataset_size))
        self.save_tfr_dataset(train_dataset, self.file_out_tfr_train)

        test_dataset = full_dataset.skip(train_size)
        val_dataset = test_dataset.skip(test_size)
        val_dataset_size = sum(1 for record in val_dataset)
        print("val_dataset_size = {}".format(val_dataset_size))
        self.save_tfr_dataset(val_dataset, self.file_out_tfr_val)

        test_dataset = test_dataset.take(test_size)
        test_dataset_size = sum(1 for record in test_dataset)
        print("test_dataset_size = {}".format(test_dataset_size))
        self.save_tfr_dataset(test_dataset, self.file_out_tfr_test)




if __name__ == "__main__":

    input_file = "/Users/c325018/ComplaintsProjects/pegasus/pegasus/data/testdata/emails_complains_pattern.tfrecords"
    file_out_tfr_train = "/Users/c325018/ComplaintsProjects/pegasus/pegasus/data/testdata/emails_complains_pattern_train.tfrecords"
    file_out_tfr_test = "/Users/c325018/ComplaintsProjects/pegasus/pegasus/data/testdata/emails_complains_pattern_test.tfrecords"
    file_out_tfr_val = "/Users/c325018/ComplaintsProjects/pegasus/pegasus/data/testdata/emails_complains_pattern_dev.tfrecords"

    percentaje_train = 0.7
    percentaje_test = 0.15
    percentaje_val = 0.15

    sd = SplittingData(input_file, percentaje_train, percentaje_test, percentaje_val,\
                       file_out_tfr_train, file_out_tfr_test, file_out_tfr_val)
    sd.run()


    #python3 train.py --params=new_params --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/complains_emails

    #python3 evaluate.py --params=new_params --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=5,beam_alpha=0.6 --model_dir=ckpt/pegasus_ckpt
    #python3 evaluate.py --params=new_params --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=5,beam_alpha=0.6 --model_dir=ckpt/pegasus_ckpt/new_params
