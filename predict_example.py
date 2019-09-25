from thainlplib import ThaiWordSegmentLabeller
import tensorflow as tf
import numpy as np

saved_model_path='saved_model'



def nonzero(a):
    return [i for i, e in enumerate(a) if e != 0]

def split(s, indices):
    return [s[i:j] for i,j in zip(indices, indices[1:]+[None])]

def sertis_tokenizer(text, saved_model_path):
    # inputs = [ThaiWordSegmentLabeller.get_input_labels(text)]
    inputs = [[ThaiWordSegmentLabeller.get_input_labels(i)] for i in text]
    print(inputs)
    lengths = [[len(i)] for i in text]
    print(lengths)
    with tf.Session() as session:
        model = tf.saved_model.loader.load(session, [tf.saved_model.tag_constants.SERVING], saved_model_path)
        signature = model.signature_def[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        graph = tf.get_default_graph()

        g_inputs = graph.get_tensor_by_name(signature.inputs['inputs'].name)
        g_lengths = graph.get_tensor_by_name(signature.inputs['lengths'].name)
        g_training = graph.get_tensor_by_name(signature.inputs['training'].name)
        g_outputs = graph.get_tensor_by_name(signature.outputs['outputs'].name)
        # y = session.run(g_outputs, feed_dict = {g_inputs: inputs, g_lengths: lengths, g_training: False})
        # y = session.run(g_outputs, feed_dict = {g_inputs: inputs, g_lengths: lengths, g_training: False})
        for i, j in enumerate(inputs):
            print(i)
            print(j)
            print(lengths[i])
            y = session.run(g_outputs, feed_dict = {g_inputs: j, g_lengths: lengths[i], g_training: False})
            print(split(text[i], nonzero(y)))
        #print(y)
        return [split(text, nonzero(y))]

w = sertis_tokenizer(['ฉันกินข้าวปลาอาหาร', 'ฉันไม่กินข้าว ปลา อาหาร'], saved_model_path)
#print(w)