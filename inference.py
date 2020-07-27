import itertools
import os
import time

from absl import logging
from pegasus.data import infeed
from pegasus.params import all_params  # pylint: disable=unused-import
from pegasus.params import estimator_utils
from pegasus.params import registry
import tensorflow as tf
from pegasus.eval import text_eval
from pegasus.ops import public_parsing_ops


tf.enable_eager_execution()

master = ""
model_dir = "./ckpt/pegasus_ckpt/aeslc"
use_tpu = False
iterations_per_loop = 1000
num_shards = 1
param_overrides = "vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=5,beam_alpha=0.6"


eval_dir = os.path.dirname(model_dir)
checkpoint_path = model_dir
checkpoint_path = tf.train.latest_checkpoint(checkpoint_path )
params = registry.get_params('aeslc_transformer')(param_overrides)
pattern = params.dev_pattern
input_fn = infeed.get_input_fn(params.parser, pattern,
                                     tf.estimator.ModeKeys.PREDICT)
parser, shapes = params.parser(mode=tf.estimator.ModeKeys.PREDICT)


estimator = estimator_utils.create_estimator(master,
                                             model_dir,
                                             use_tpu,
                                             iterations_per_loop,
                                             num_shards, params)

_SPM_VOCAB = 'ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model'
encoder = public_parsing_ops.create_text_encoder("sentencepiece",
                                                     _SPM_VOCAB)



input_text1 = "hello this is a first text"
target1 = "first text"


input_text2 = "Eighteen sailors were injured after an explosion and fire on board a ship at the US Naval Base in San Diego, US Navy officials said.The sailors on the USS Bonhomme Richard had 'minor injuries' from the fire and were taken to a hospital, Lt. Cmdr. Patricia Kreuzberger told CNN."
target2 = "18 sailors injured after an explosion and fire on a naval ship in San Diego"


def input_function(params):
    dataset = tf.data.Dataset.from_tensor_slices({"inputs":[input_text1, input_text2],"targets":[target1, target2]}).map(parser)
    dataset = dataset.unbatch()
    dataset = dataset.padded_batch(
        params["batch_size"],
        padded_shapes=shapes,
        drop_remainder=True)
    return dataset

predictions = estimator.predict(
          input_fn=input_function, checkpoint_path=checkpoint_path)

for i in predictions:
    print(text_eval.ids2str(encoder, i['outputs'], None))

# Ouput - "The USS Bonhomme Richard had 'minor injuries' from the fire and were taken to a hospital ."