from Bio import SeqIO
from CRF import CRF
from crf_loss import ConditionalRandomFieldLoss
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras import layers
import random
import numpy as np
import tensorflow as tf
from keras_ordered_neurons import ONLSTM

from constants import Kingdom, SPType, PositionSpecificLetter
from data_transform import (
    classes_sequence_to_letters,
    input_from_sequence,
    classes_sequence_from_ann_sequence,
)

kingdoms_encoder = preprocessing.OneHotEncoder()
kingdoms_encoder.fit(np.array(Kingdom.values()).reshape((len(Kingdom.values()), 1)))

sp_types_encoder = preprocessing.OneHotEncoder()
sp_types_encoder.fit(np.array(SPType.values()).reshape((len(SPType.values()), 1)))

position_specific_classes_enc = preprocessing.LabelEncoder()
position_specific_classes_enc.fit(
    np.array(PositionSpecificLetter.values()).reshape((len(PositionSpecificLetter.values()), 1))
)

# Callback that prints out the expected and predicted sequence of position-specific classes
# for a number of samples
class SamplePredictionsCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        data_sequences,
        data_kingdoms,
        data_class_sequences,
        data_sp_types,
        data_ids,
        position_specific_classes_enc,
        sp_types_enc,
        sample_size=5,
    ):
        self.data_sequences = data_sequences
        self.data_kingdoms = data_kingdoms
        self.data_class_sequences = data_class_sequences
        self.data_sp_types = data_sp_types
        self.data_ids = data_ids

        self.position_specific_classes_enc = position_specific_classes_enc
        self.sp_types_enc = sp_types_enc

        self.sample_size = sample_size

        super().__init__()

    def on_test_begin(self, a):
        for sample_i in random.sample(range(0, len(self.data_sequences) - 1), self.sample_size):
            input_sequence = self.data_sequences[sample_i]
            input_kingdom = self.data_kingdoms[sample_i]
            expected_class_sequence = self.data_class_sequences[sample_i]
            expected_sp_type = self.data_sp_types[sample_i]
            vld_id = self.data_ids[sample_i]

            pred = self.model.predict(
                {
                    "aa_sequence": np.expand_dims(input_sequence, axis=0),
                    "kingdom": np.expand_dims(input_kingdom, axis=0),
                }
            )
            sequence_pred = pred[0][0]
            sp_type_pred = pred[1][0]

            print("ID:", vld_id)
            print(
                "expected SEQ  :",
                classes_sequence_to_letters(
                    expected_class_sequence, self.position_specific_classes_enc
                ),
            )
            print(
                "prediction SEQ:",
                classes_sequence_to_letters(sequence_pred, self.position_specific_classes_enc),
            )
            print(
                "expected   SP_TYPE:", self.sp_types_enc.inverse_transform([expected_sp_type])[0][0]
            )
            print("prediction SP_TYPE:", self.sp_types_enc.inverse_transform([sp_type_pred])[0][0])


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
        output_sp_type_single = np.squeeze(
            np.array(sp_types_encoder.transform([[sp_type]]).todense())
        )
        output_sp_type.append(output_sp_type_single)

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
    bi_lstm = layers.Bidirectional(
        ONLSTM(units=64, chunk_size=4, recurrent_dropconnect=0.2, return_sequences=True),
        merge_mode="mul",
    )(cnn1_plus_kingdom)
    cnn2 = layers.Dropout(0.1)(layers.Conv1D(64, (5), padding="causal", activation="relu")(bi_lstm))
    # 9 = "matching the number of position-specific classes"
    cnn3 = layers.Dropout(0.1)(layers.Conv1D(9, (1), padding="causal", activation="relu")(cnn2))
    crf_raw = CRF(9, use_kernel=False, name="crf")
    crf = crf_raw(cnn3)
    crf_loss = ConditionalRandomFieldLoss(crf_raw)

    sp_type_out = layers.Dense(4, activation="relu", name="sp_type")(crf)

    outputs = {"crf": crf, "sp_type": sp_type_out}
    model = keras.Model(inputs=[sequence_input, kingdom_input], outputs=outputs)

    return (model, crf_loss)


def train_model(model, crf_loss):
    model.summary()

    opt = keras.optimizers.SGD(0.005)
    model.compile(
        optimizer=opt,
        loss={
            "crf": crf_loss,
            "sp_type": tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        },
        metrics=["acc"],
    )

    (
        vld_input_sequences,
        vld_input_kingdoms,
        vld_output_class_sequence,
        vld_output_sp_types,
        vld_ids,
    ) = build_set(max_length=0, dataset_path="benchmark_set.fasta")
    (input_sequences, input_kingdoms, output_class_sequence, output_sp_types, _) = build_set(
        max_length=0
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="./tf_ckpt/mymodel_{epoch}.h5", save_best_only=True, verbose=1
        ),
        SamplePredictionsCallback(
            vld_input_sequences,
            vld_input_kingdoms,
            vld_output_class_sequence,
            vld_output_sp_types,
            vld_ids,
            position_specific_classes_enc,
            sp_types_encoder,
        ),
    ]
    model.fit(
        {"aa_sequence": input_sequences, "kingdom": input_kingdoms},
        {"crf": output_class_sequence, "sp_type": output_sp_types},
        validation_data=(
            {"aa_sequence": vld_input_sequences, "kingdom": vld_input_kingdoms},
            {"crf": vld_output_class_sequence, "sp_type": vld_output_sp_types},
        ),
        epochs=100,
        batch_size=128,
        verbose=1,
        callbacks=callbacks,
    )


(model, crf_loss) = build_model()
train_model(model, crf_loss)
