from Bio import SeqIO
from CRF import CRF
from crf_loss import ConditionalRandomFieldLoss
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras import layers, models
import random
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from constants import kingdoms, sp_types, position_specific_classes
from data_transform import (
    classes_sequence_to_letters,
    input_from_sequence,
    classes_sequence_from_ann_sequence,
)

kingdoms_encoder = preprocessing.OneHotEncoder()
kingdoms_encoder.fit(np.array(kingdoms).reshape((len(kingdoms), 1)))

sp_types_encoder = preprocessing.OneHotEncoder()
sp_types_encoder.fit(np.array(sp_types).reshape((len(sp_types), 1)))

position_specific_classes_enc = preprocessing.LabelEncoder()
position_specific_classes_enc.fit(
    np.array(position_specific_classes).reshape((len(position_specific_classes), 1))
)

# Callback that prints out the expected and predicted sequence of position-specific classes
# for a number of samples
class SamplePredictionsCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        data_sequences,
        data_kingdoms,
        data_class_sequences,
        data_ids,
        position_specific_classes_enc,
        sample_size=5,
    ):
        self.data_sequences = data_sequences
        self.data_kingdoms = data_kingdoms
        self.data_class_sequences = data_class_sequences
        self.data_ids = data_ids

        self.position_specific_classes_enc = position_specific_classes_enc

        self.sample_size = sample_size

        super().__init__()

    def on_test_begin(self, a):
        for sample_i in random.sample(range(0, len(self.data_sequences) - 1), self.sample_size):
            input_sequence = self.data_sequences[sample_i]
            input_kingdom = self.data_kingdoms[sample_i]
            expected_class_sequence = self.data_class_sequences[sample_i]
            vld_id = self.data_ids[sample_i]

            pred = self.model.predict(
                {
                    "aa_sequence": np.expand_dims(input_sequence, axis=0),
                    "kingdom": np.expand_dims(input_kingdom, axis=0),
                }
            )[0]
            print("ID:", vld_id)
            print(
                "expected  :",
                classes_sequence_to_letters(
                    expected_class_sequence, self.position_specific_classes_enc
                ),
            )
            print(
                "prediction:", classes_sequence_to_letters(pred, self.position_specific_classes_enc)
            )


def build_set(max_length=0, dataset_path="train_set.fasta"):

    input_tensors = []
    input_kingdoms = []
    output_class_sequence = []
    output_sp_type = []
    ids = []
    for record in SeqIO.parse(dataset_path, "fasta"):
        if max_length:
            if len(input_tensors) == max_length:
                continue

        # TODO: skipping non 70 length sequence for now; ISSUE #1
        if int(len(record)) != 140:
            continue

        half_len = int(len(record) / 2)

        sequence = record[0:half_len]
        ann_sequence = record[half_len : int(len(record))]

        feature_parts = record.id.split("|")
        (uniprot_id, kingdom, sp_type, partition) = feature_parts

        ids.append(uniprot_id)
        input_tensors.append(input_from_sequence(sequence.seq))
        input_kingdoms.append(
            np.repeat(kingdoms_encoder.transform([[kingdom]]).todense(), 70, axis=0)
        )
        output_class_sequence.append(
            classes_sequence_from_ann_sequence(ann_sequence.seq, position_specific_classes_enc)
        )
        output_sp_type.append(np.squeeze(sp_types_encoder.transform([[sp_type]]).todense()))

    return (
        np.array(input_tensors),
        np.array(input_kingdoms),
        np.array(output_class_sequence),
        np.array(output_sp_type),
        ids,
    )


def build_model():
    sequence_input = keras.Input(shape=(70, 20), name="aa_sequence")
    kingdom_input = keras.Input(shape=(70, 4), name="kingdom")
    cnn1 = layers.Dropout(0.1)(
        layers.Conv1D(32, (3), padding="causal", activation="relu")(sequence_input)
    )
    cnn1_plus_kingdom = layers.concatenate([cnn1, kingdom_input])
    bi_lstm = layers.Bidirectional(layers.LSTM(64, return_sequences=True), merge_mode="mul")(
        cnn1_plus_kingdom
    )
    cnn2 = layers.Dropout(0.1)(layers.Conv1D(64, (5), padding="causal", activation="relu")(bi_lstm))
    # 9 = "matching the number of position-specific classes"
    cnn3 = layers.Dropout(0.1)(layers.Conv1D(9, (1), padding="causal", activation="relu")(cnn2))
    crf_raw = CRF(9, use_kernel=False, name="crf")
    crf = crf_raw(cnn3)
    crf_loss = ConditionalRandomFieldLoss(crf_raw)

    outputs = {"crf": crf}
    model = keras.Model(inputs=[sequence_input, kingdom_input], outputs=outputs)

    return (model, crf_loss)


def train_model(model, crf_loss):
    model.summary()

    opt = keras.optimizers.SGD(0.005)
    model.compile(optimizer=opt, loss={"crf": crf_loss}, metrics=["acc"])

    (vld_input_sequences, vld_input_kingdoms, vld_output_class_sequence, _, vld_ids) = build_set(
        max_length=0, dataset_path="benchmark_set.fasta"
    )
    (input_sequences, input_kingdoms, output_class_sequence, _, _) = build_set(max_length=0)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="./tf_ckpt/mymodel_{epoch}.h5", save_best_only=True, verbose=1
        ),
        SamplePredictionsCallback(
            vld_input_sequences,
            vld_input_kingdoms,
            vld_output_class_sequence,
            vld_ids,
            position_specific_classes_enc,
        ),
    ]
    model.fit(
        {"aa_sequence": input_sequences, "kingdom": input_kingdoms},
        {"crf": output_class_sequence},
        validation_data=(
            {"aa_sequence": vld_input_sequences, "kingdom": vld_input_kingdoms},
            {"crf": vld_output_class_sequence},
        ),
        epochs=100,
        batch_size=128,
        verbose=1,
        callbacks=callbacks,
    )


(model, crf_loss) = build_model()
train_model(model, crf_loss)
