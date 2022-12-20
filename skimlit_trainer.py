
from typing import Dict, List, Text

import os
import glob
from absl import logging

import datetime
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_hub as hub
from tfx import v1 as tfx
from tfx_bsl.public import tfxio
from tensorflow_transform import TFTransformOutput
import string
from tensorflow.keras import layers

alphabet = string.ascii_lowercase + string.digits + string.punctuation


_LABEL_KEY = 'target_int'
NUM_CHAR_TOKENS = len(alphabet) + 2
output_seq_char_len= 290



def _input_fn(file_pattern: List[Text],
              data_accessor: tfx.components.DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 64) -> tf.data.Dataset:
    
    dataset_ = data_accessor.tf_dataset_factory(
                                        file_pattern,
                                        tfxio.TensorFlowDatasetOptions(batch_size=batch_size, label_key=_LABEL_KEY),
                                        tf_transform_output.transformed_metadata.schema)
    return dataset_   
                                 
    

def _get_tf_examples_serving_signature(model, tf_transform_output):
  """Returns a serving signature that accepts `tensorflow.Example`."""
  # We need to track the layers in the model in order to save it.
  # TODO(b/162357359): Revise once the bug is resolved.


  model.tft_layer_inference = tf_transform_output.transform_features_layer()

  @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')])
  def serve_tf_examples_fn(serialized_tf_example):
    """Returns the output to be used in the serving signature."""

    raw_feature_spec = tf_transform_output.raw_feature_spec()
 
    # Remove label feature since these will not be present at serving time.
    raw_feature_spec.pop(_LABEL_KEY)
    raw_feature_spec.pop('target')
    
    raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
    transformed_features = model.tft_layer_inference(raw_features)
    logging.info('serve_transformed_features = %s', transformed_features)

    outputs = model(transformed_features)

    # TODO(b/154085620): Convert the predicted labels from the model using a
    # reverse-lookup (opposite of transform.py).

    return {'outputs': outputs}

  return serve_tf_examples_fn


def _get_transform_features_signature(model, tf_transform_output):
  """Returns a serving signature that applies tf.Transform to features."""
  
  # We need to track the layers in the model in order to save it.
  # TODO(b/162357359): Revise once the bug is resolved.

  model.tft_layer_eval = tf_transform_output.transform_features_layer()

  @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')])
  def transform_features_fn(serialized_tf_example):
    """Returns the transformed_features to be fed as input to evaluator."""

    raw_feature_spec = tf_transform_output.raw_feature_spec()
    raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
  
    transformed_features = model.tft_layer_eval(raw_features)
    logging.info('eval_transformed_features = %s', transformed_features)
  
    return transformed_features

  return transform_features_fn


def export_serving_model(tf_transform_output, model, output_dir):
  """Exports a keras model for serving.
  Args:
    tf_transform_output: Wrapper around output of tf.Transform.
    model: A keras model to export for serving.
    output_dir: A directory where the model will be exported to.
  """

  # The layer has to be saved to the model for keras tracking purpases.
  
  model.tft_layer = tf_transform_output.transform_features_layer()

  signatures = {
      'serving_default':
          _get_tf_examples_serving_signature(model, tf_transform_output),
      'transform_features':
          _get_transform_features_signature(model, tf_transform_output),
  }

  model.save(output_dir, save_format='tf', signatures=signatures)


tf_hub_embedding_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                            trainable=False,
                                            name="universal_sentence_encoder")

char_vectorizer = layers.TextVectorization(max_tokens=NUM_CHAR_TOKENS,  
                                    output_sequence_length=output_seq_char_len,
                                    standardize="lower_and_strip_punctuation",
                                    name="char_vectorizer")


char_embed = layers.Embedding(input_dim=NUM_CHAR_TOKENS,
                              output_dim=25, 
                              mask_zero=False, 
                              name="char_embed")

def model_builder():
    
    token_inputs = layers.Input(shape=[], dtype="string", name="token_inputs")
    token_embeddings = tf_hub_embedding_layer(token_inputs)
    token_outputs = layers.Dense(128, activation="relu")(token_embeddings)
    token_model = tf.keras.Model(inputs=token_inputs,
                                outputs=token_outputs)

    # 2. Char inputs
    char_inputs = layers.Input(shape=(1,), dtype="string", name="char_inputs")
    char_vectors = char_vectorizer(char_inputs)
    char_embeddings = char_embed(char_vectors)
    char_bi_lstm = layers.Bidirectional(layers.LSTM(32))(char_embeddings)
    char_model = tf.keras.Model(inputs=char_inputs,
                                outputs=char_bi_lstm)

    # 3. Line numbers inputs
    line_number_inputs = layers.Input(shape=(15,), dtype=tf.int32, name="line_number_input")
    x = layers.Dense(32, activation="relu")(line_number_inputs)

    line_number_model = tf.keras.Model(inputs=line_number_inputs,
                                    outputs=x)

    # 4. Total lines inputs
    total_lines_inputs = layers.Input(shape=(20,), dtype=tf.int32, name="total_lines_input")
    y = layers.Dense(32, activation="relu")(total_lines_inputs)
    total_line_model = tf.keras.Model(inputs=total_lines_inputs,
                                    outputs=y)

    # 5. Combine token and char embeddings into a hybrid embedding
    combined_embeddings = layers.Concatenate(name="token_char_hybrid_embedding")([token_model.output, 
                                                                                char_model.output])
    z = layers.Dense(256, activation="relu")(combined_embeddings)
    z = layers.Dropout(0.5)(z)

    # 6. Combine positional embeddings with combined token and char embeddings into a tribrid embedding
    z = layers.Concatenate(name="token_char_positional_embedding")([line_number_model.output,
                                                                    total_line_model.output,
                                                                    z])


    # 7. Create output layer
    output_layer = layers.Dense(5, activation="softmax", name="output_layer")(z)

    # 8. Put together model
    model_5 = tf.keras.Model(inputs=[line_number_model.input,
                                    total_line_model.input,
                                    token_model.input, 
                                    char_model.input],
                            outputs=output_layer)


    model_5.compile(loss='categorical_crossentropy',
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

    return model_5


    
def run_fn(fn_args: tfx.components.FnArgs) -> None:
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir = fn_args.model_run_dir, update_freq='batch'
    )
    
    es = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', mode='max', verbose=1, patience=10)
    mc = tf.keras.callbacks.ModelCheckpoint(fn_args.serving_model_dir, monitor='val_categorical_accuracy', mode='max', verbose=1, save_best_only=True)
    

    # Load the transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    
    # Create batches of data
    train_set = _input_fn(fn_args.train_files,fn_args.data_accessor, tf_transform_output)
    val_set = _input_fn(fn_args.eval_files,fn_args.data_accessor, tf_transform_output)
    
    
    vectorize_dataset = train_set.map(lambda f, l: f['char_inputs']).unbatch()
    char_vectorizer.adapt(vectorize_dataset.take(1000))

    # Build the model
    model = model_builder()

    
    # Train the model
    model.fit(train_set,
            validation_data = val_set,
            callbacks = [tensorboard_callback, es, mc],
            steps_per_epoch = 1000, 
            validation_steps= 1000,
            epochs=1)
    export_serving_model(tf_transform_output, model, fn_args.serving_model_dir)
    
