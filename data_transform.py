import Bio
import numpy as np
import cachetools
from Bio.SubsMat import MatrixInfo

from constants import position_specific_letters


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
