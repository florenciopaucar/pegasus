import pandas as pd
import tensorflow as tf
import numpy as np

# Load
read_dictionary = np.load('utils/dic_data.npy', allow_pickle='TRUE').item()
print(len(read_dictionary["inputs"]))
print(len(read_dictionary["targets"]))

df = pd.DataFrame(read_dictionary)


header = ["inputs", "targets"]

df.to_csv('utils/output.csv', columns=header, index=False)
df.head(10)
csv = pd.read_csv("utils/output.csv").values
with tf.io.TFRecordWriter("pegasus/data/testdata/emails_complains_pattern.tfrecords") as writer:
    for row in csv:
        inputs, targets = str(row[:-1]), str(row[-1])
        #print("targets: "+ str(targets))
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "inputs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[inputs[0].encode('utf-8')])),
                    "targets": tf.train.Feature(bytes_list=tf.train.BytesList(value=[targets.encode('utf-8')])),
                }
            )
        )
        writer.write(example.SerializeToString())


#EVALUATE THE DATA
#python3 evaluate.py --params=new_params --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=5,beam_alpha=0.6 --model_dir=ckpt/pegasus_ckpt
