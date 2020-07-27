import numpy as np
import tensorflow as tf

from pegasus.eval import text_eval
from pegasus.ops import public_parsing_ops
from pegasus.params import estimator_utils
from pegasus.params import registry

tf.enable_eager_execution()


class InferencePegasus():

    def __init__(self, param_overrides, model_dir, vocab_filename, encoder_type, params_transformer,
                 test_dict_dataset_path):
        self.param_overrides = param_overrides
        self.model_dir = model_dir
        self.vocab_filename = vocab_filename
        self.encoder_type = encoder_type
        self.params_transformer = params_transformer
        self.test_dict_dataset_path = test_dict_dataset_path

        self.master = ""
        self.use_tpu = False
        self.iterations_per_loop = 1000
        self.num_shards = 1

    def run(self):
        checkpoint_path = self.model_dir
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        params = registry.get_params(self.params_transformer)(self.param_overrides)
        parser, shapes = params.parser(mode=tf.estimator.ModeKeys.PREDICT)
        estimator = estimator_utils.create_estimator(self.master,
                                                     self.model_dir,
                                                     self.use_tpu,
                                                     self.iterations_per_loop,
                                                     self.num_shards, params)
        encoder = public_parsing_ops.create_text_encoder(self.encoder_type, self.vocab_filename)

        def input_function(params):
            input_text1 = "hello this is a first text"
            target1 = "first text"
            input_text2 = "Eighteen sailors were injured after an explosion and fire on board a ship at the US Naval Base in San Diego, US Navy officials said.The sailors on the USS Bonhomme Richard had 'minor injuries' from the fire and were taken to a hospital, Lt. Cmdr. Patricia Kreuzberger told CNN."
            target2 = "18 sailors injured after an explosion and fire on a naval ship in San Diego"
            read_dictionary_data = np.load(self.test_dict_dataset_path, allow_pickle='TRUE').item()
            # dataset = tf.data.Dataset.from_tensor_slices({"inputs":[input_text1, input_text2],"targets":[target1, target2]}).map(parser)

            dataset = tf.data.Dataset.from_tensor_slices(read_dictionary_data).map(parser)
            dataset = dataset.unbatch()
            dataset = dataset.padded_batch(
                params["batch_size"],
                padded_shapes=shapes,
                drop_remainder=True)
            return dataset

        predictions = estimator.predict(input_fn=input_function, checkpoint_path=checkpoint_path)
        for i in predictions:
            print(
                "=======================================================================================================================================")
            print("inputs: " + text_eval.ids2str(encoder, i['inputs'], None))
            print("targets: " + text_eval.ids2str(encoder, i['targets'], None))
            print("outputs: " + text_eval.ids2str(encoder, i['outputs'], None))


if __name__ == "__main__":
    param_overrides = "vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=5,beam_alpha=0.6"
    model_dir = "ckpt/pegasus_ckpt/aeslc"
    vocab_filename = 'ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model'
    encoder_type = "sentencepiece"
    params_transformer = 'aeslc_transformer'
    test_dict_dataset_path = "utils/dic_data.npy"

    ip = InferencePegasus(param_overrides, model_dir, vocab_filename, encoder_type, params_transformer,
                          test_dict_dataset_path)
    ip.run()
