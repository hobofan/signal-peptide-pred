# Taken from https://github.com/tensorflow/addons/blob/674267ff9cac7ab3f108d78b830660a2aeed4376

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementing Conditional Random Field loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from CRF import CRF
from tensorflow_addons.utils import keras_utils


#  @keras_utils.register_keras_custom_object
class ConditionalRandomFieldLoss(object):
    def get_config(self):
        return {}

    def __call__(self, y_true, y_pred, sample_weight=None):
        crf_layer = y_pred._keras_history[0]

        # check if last layer is CRF
        if not isinstance(crf_layer, CRF):
            raise ValueError("Last layer must be CRF for use {}.".format(self.__class__.__name__))

        loss_vector = crf_layer.get_loss(y_true, y_pred)

        return tf.keras.backend.mean(loss_vector)
