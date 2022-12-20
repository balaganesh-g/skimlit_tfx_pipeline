import sys
import absl
import tensorflow as tf
import tensorflow_transform as tft 
import constant
import numpy as np
from sklearn.preprocessing import OneHotEncoder



if 'google.colab' in sys.modules:  # Testing to see if we're doing development
  import importlib
  importlib.reload(constant)


_ONE_HOT_FEATURES = constant.ONE_HOT_FEATURES
_CAT_FEATURES = constant.ONE_HOT_FEATURES
_TARGET_FEATURE = constant.TARGET_FEATURE
_OOV_SIZE = constant.OOV_SIZE
_VOCAB_SIZE = constant.VOCAB_SIZE


def _make_one_hot(x,key):

  one_hot_encoded = tf.one_hot(
      x,
      depth=20,
      on_value=1.0,
      off_value=0.0)
  return tf.reshape(one_hot_encoded,[-1,20])


def _make_one_hot_line_number(x,key):

  one_hot_encoded = tf.one_hot(
      x,
      depth=15,
      on_value=1.0,
      off_value=0.0)
  return tf.reshape(one_hot_encoded,[-1,15])


def _fill_in_missing(x):
  """Replace missing values in a SparseTensor.
  Fills in missing values of `x` with '' or 0, and converts to a dense tensor.
  Args:
    x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
      in the second dimension.
  Returns:
    A rank 1 tensor where missing values of `x` have been filled in.
  """
  if not isinstance(x, tf.sparse.SparseTensor):
    return x

  default_value = '' if x.dtype == tf.string else 0
  return tf.squeeze(
      tf.sparse.to_dense(
          tf.sparse.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
          default_value),
      axis=1)


def _make_train_char(x):
    s = tf.strings.regex_replace(x, ' ', '')
    s = tf.strings.regex_replace(s, '', ' ')
    s = tf.strings.strip(s)
    return x ,s


def _make_one_hot_target(x):

  one_hot_encoded = tf.one_hot(
      x,
      depth=5,
      on_value=1.0,
      off_value=0.0)
  return tf.reshape(one_hot_encoded,[-1,5])


def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.
  Args:
    inputs: map from feature keys to raw not-yet-transformed features.
  Returns:
    Map from string feature key to transformed feature operations.
  """
  outputs = {}

  total_lines = inputs['total_lines']
  onehot_total_lines = _make_one_hot(_fill_in_missing(total_lines), 'total_lines')

  line_number = inputs['line_number']
  line_numbers = _make_one_hot_line_number(_fill_in_missing(line_number), 'line_number')

  text = inputs['text']
  train_sentence,train_chars = _make_train_char(_fill_in_missing(text))

  target = inputs['target_int']
  target = _make_one_hot_target(_fill_in_missing(target))

  return  {
      'line_number_input':line_numbers,
      'total_lines_input':onehot_total_lines,
      'token_inputs':train_sentence,
      'char_inputs':  train_chars,
      'target_int': target
  }  
