from Bio import SeqIO
from Bio.SubsMat import MatrixInfo
from CRF import CRF
from crf_loss import ConditionalRandomFieldLoss
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import random
import Bio
import cachetools
import functools
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

kingdoms = ["EUKARYA", "ARCHAEA", "NEGATIVE", "POSITIVE"]
sp_types = ["NO_SP", "SP", "TAT", "LIPO"]
ann_letters = [
    # inner
    "I",
    # outer
    "O",
    # tm_in_out | tm_out_in
    "M",
    # signal_SecSP1 | cs_SP1
    "S",
    # signal_SecSP2 | cs_SP2
    "L",
    # signal_TatSP1 | cs_SP1
    "T",
]
position_specific_classes = [
    # S
    "signal_SecSP1",
    # Z
    "signal_SecSP2",
    # T
    "signal_TatSP1",
    # C
    "cs_SP1",
    # K
    "cs_SP2",
    # O
    "outer",
    # I
    "inner",
    # L (Leaving)
    "tm_in_out",
    # E (Entering)
    "tm_out_in",
]
position_specific_letters = {
    "signal_SecSP1": "S",
    "signal_SecSP2": "Z",
    "signal_TatSP1": "T",
    "cs_SP1": "C",
    "cs_SP2": "K",
    "outer": "O",
    "inner": "I",
    "tm_in_out": "L",
    "tm_out_in": "E",
}

kingdoms_encoder = preprocessing.OneHotEncoder()
kingdoms_encoder.fit(np.array(kingdoms).reshape((len(kingdoms), 1)))

position_specific_classes_enc = preprocessing.LabelEncoder()
position_specific_classes_enc.fit(
    np.array(position_specific_classes).reshape((len(position_specific_classes), 1))
)


def get_replacement_score(a, b, replacement_matrix):
    if (a, b) not in replacement_matrix:
        return replacement_matrix[b, a]
    return replacement_matrix[a, b]


def array_get(arr, i):
    try:
        return arr[i]
    except IndexError:
        return None


def hash_sequence(sequence, other=None):
    return hash(str(sequence))


@cachetools.cached({}, key=hash_sequence)
def input_from_sequence(sequence):
    replacement_matrix = MatrixInfo.blosum62

    replacement_probabilities = []
    for i in range(70):
        letter_replacement_probabilities = []
        try:
            used_letter = sequence[i]
            for replacement_letter in Bio.Alphabet.IUPAC.IUPACProtein.letters:
                letter_replacement_probabilities.append(
                    get_replacement_score(used_letter, replacement_letter, replacement_matrix)
                )
        except IndexError:
            # Zero pad short sequences
            letter_replacement_probabilities = np.repeat(0, 20)

        replacement_probabilities.append(letter_replacement_probabilities)

    return np.array(replacement_probabilities)


@cachetools.cached({}, key=hash_sequence)
def classes_sequence_from_ann_sequence(sequence, enc):
    print("SEQ", sequence)
    #  if sequence == "IIIIIIIIIIIMMMMMMMMMMMMMMMMMMMMMOOOOOMMMMMMMMMMMMMMMMMIIIIIIIIIIIIIIII":
    #  print("SEQ_!", sequence)

    classes_sequence = []
    prev_inner_outer = None
    for i in range(70):
        letter = array_get(sequence, i)
        if letter is None:
            #  placeholder_encoding = [np.zeros(len(enc.classes_[0]))]
            placeholder_encoding = [0]
            classes_sequence.append(placeholder_encoding)
            continue

        prev_letter = sequence[i - 1] if i > 0 else None
        next_letter = sequence[i + 1] if i + 1 < len(sequence) else None

        position_specific_class = None

        if letter == "I":
            position_specific_class = "inner"
            prev_inner_outer = "I"
        elif letter == "O":
            position_specific_class = "outer"
            prev_inner_outer = "O"
        elif letter == "M":
            if prev_letter == "M":
                if prev_inner_outer == "I":
                    position_specific_class = "tm_in_out"
                elif prev_inner_outer == "O":
                    position_specific_class = "tm_out_in"
            elif prev_letter == "I" or prev_inner_outer == "I":
                position_specific_class = "tm_in_out"
            elif prev_letter == "O" or prev_inner_outer == "O":
                position_specific_class = "tm_out_in"
        elif letter == "S":
            if next_letter == "S":
                position_specific_class = "signal_SecSP1"
            else:
                position_specific_class = "cs_SP1"
        elif letter == "L":
            if next_letter == "L":
                position_specific_class = "signal_SecSP2"
            else:
                position_specific_class = "cs_SP2"
        elif letter == "T":
            if next_letter == "T":
                position_specific_class = "signal_TatSP1"
            else:
                position_specific_class = "cs_SP1"

        if position_specific_class is None:
            print("Unexpected case", prev_letter, letter, next_letter)

        transformed = enc.transform([position_specific_class])
        classes_sequence.append(transformed)

    seq_tensor = np.array(classes_sequence).reshape((70))
    #  if sequence == "IIIIIIIIIIIMMMMMMMMMMMMMMMMMMMMMOOOOOMMMMMMMMMMMMMMMMMIIIIIIIIIIIIIIII":
    #  print("SE!", classes_sequence_to_letters(seq_tensor, enc))

    return seq_tensor


def classes_sequence_to_letters(seq_tensor, enc):
    letters = []
    for n in seq_tensor:
        klass = enc.inverse_transform([n])[0]
        letter = position_specific_letters[klass]
        letters.append(letter)

    return "".join(letters)


def build_set(max_length=0, dataset_path="train_set.fasta"):

    input_tensors = []
    input_kingdoms = []
    output_class_sequence = []
    ids = []
    for record in SeqIO.parse(dataset_path, "fasta"):
        if max_length:
            if len(input_tensors) == max_length:
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

    return (np.array(input_tensors), np.array(input_kingdoms), np.array(output_class_sequence), ids)


def build_old_model():
    model = models.Sequential()
    model.add()
    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True), merge_mode="mul"))
    model.add(layers.Conv1D(64, (5), activation="relu"))
    model.add(
        layers.Conv1D(9, (1), activation="relu")
    )  # 9 = "matching the number of position-specific classes"
    model.add(tfa.text.crf.CrfDecodeForwardRnnCell([70]))

    model.summary()


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
    crf = CRF(9, use_kernel=False, name="crf")(cnn3)

    outputs = {"crf": crf}
    model = keras.Model(inputs=[sequence_input, kingdom_input], outputs=outputs)

    model.summary()

    opt = keras.optimizers.SGD(0.005)
    model.compile(optimizer=opt, loss={"crf": ConditionalRandomFieldLoss()}, metrics=["acc"])

    (vld_input_sequences, vld_input_kingdoms, vld_output_class_sequence, vld_ids) = build_set(
        max_length=0, dataset_path="benchmark_set.fasta"
    )
    (input_sequences, input_kingdoms, output_class_sequence, _) = build_set(max_length=0)

    def sample_predictions(epoch, logs):
        sample_size = 5
        for sample_i in random.sample(range(0, len(vld_input_sequences) - 1), sample_size):
            input_sequence = vld_input_sequences[sample_i]
            input_kingdom = vld_input_kingdoms[sample_i]
            expected_class_sequence = vld_output_class_sequence[sample_i]
            vld_id = vld_ids[sample_i]

            pred = model.predict(
                {
                    "aa_sequence": np.expand_dims(input_sequence, axis=0),
                    "kingdom": np.expand_dims(input_kingdom, axis=0),
                }
            )[0]
            print("ID:", vld_id)
            print(
                "expected  :",
                classes_sequence_to_letters(expected_class_sequence, position_specific_classes_enc),
            )
            print("prediction:", classes_sequence_to_letters(pred, position_specific_classes_enc))

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="./tf_ckpt/mymodel_{epoch}.h5", save_best_only=True, verbose=1
        ),
        keras.callbacks.LambdaCallback(on_epoch_end=sample_predictions),
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


build_model()
