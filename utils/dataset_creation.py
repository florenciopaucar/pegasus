import pandas as pd
import tensorflow as tf
import numpy as np


class DatasetCreation():

    def __init__(self, input_dictionay, output_tfr_file, output_dictionary_csv):
        self.input_dictionay = input_dictionay
        self.output_tfr_file = output_tfr_file
        self.output_dictionary_csv = output_dictionary_csv

    def run(self):
        read_dictionary = np.load(self.input_dictionay, allow_pickle='TRUE').item()
        dframe = pd.DataFrame(read_dictionary)
        header = ["inputs", "targets"]
        print(len(read_dictionary["inputs"]))
        print(len(read_dictionary["targets"]))
        dframe.to_csv(output_dictionary_csv, columns=header, index=False)
        csv = pd.read_csv(output_dictionary_csv).values
        with tf.io.TFRecordWriter(output_tfr_file) as writer:
            for row in csv:
                inputs, targets = str(row[:-1]), str(row[-1])
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "inputs": tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[inputs.encode('utf-8')])),
                            "targets": tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[targets.encode('utf-8')])),
                        }
                    )
                )
                writer.write(example.SerializeToString())


if __name__ == "__main__":
    input_dictionay = "utils/dic_data.npy"
    output_dictionary_csv = "utils/output.csv"
    output_tfr_file = "pegasus/data/testdata/emails_complains_pattern.tfrecords"
    dc = DatasetCreation(input_dictionay, output_tfr_file, output_dictionary_csv)
    dc.run()
