import Bio
import numpy as np
import cachetools
from Bio.SubsMat import MatrixInfo

from constants import PositionSpecificLetter, AnnotationLetter


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
        letter = AnnotationLetter(letter)

        if letter == AnnotationLetter.INNER:
            position_specific_class = PositionSpecificLetter.INNER
            prev_inner_outer = AnnotationLetter.INNER
        elif letter == AnnotationLetter.OUTER:
            position_specific_class = PositionSpecificLetter.OUTER
            prev_inner_outer = AnnotationLetter.OUTER
        elif letter == AnnotationLetter.TRANSMEMBRANE:
            if prev_letter == AnnotationLetter.TRANSMEMBRANE:
                if prev_inner_outer == AnnotationLetter.INNER:
                    position_specific_class = PositionSpecificLetter.TRANSMEMBRANE_IN_OUT
                elif prev_inner_outer == AnnotationLetter.OUTER:
                    position_specific_class = PositionSpecificLetter.TRANSMEMBRANE_OUT_IN
            elif (
                prev_letter == AnnotationLetter.INNER or prev_inner_outer == AnnotationLetter.INNER
            ):
                position_specific_class = PositionSpecificLetter.TRANSMEMBRANE_IN_OUT
            elif (
                prev_letter == AnnotationLetter.OUTER or prev_inner_outer == AnnotationLetter.OUTER
            ):
                position_specific_class = PositionSpecificLetter.TRANSMEMBRANE_OUT_IN
        elif letter == AnnotationLetter.SIGNAL_SEC_SP1:
            if next_letter == AnnotationLetter.SIGNAL_SEC_SP1:
                position_specific_class = PositionSpecificLetter.SIGNAL_SEC_SP1
            else:
                position_specific_class = PositionSpecificLetter.CLEAVAGE_SITE_SP1
        elif letter == AnnotationLetter.SIGNAL_SEC_SP2:
            if next_letter == AnnotationLetter.SIGNAL_SEC_SP2:
                position_specific_class = PositionSpecificLetter.SIGNAL_SEC_SP2
            else:
                position_specific_class = PositionSpecificLetter.CLEAVAGE_SITE_SP2
        elif letter == AnnotationLetter.SIGNAL_TAT_SP1:
            if next_letter == AnnotationLetter.SIGNAL_TAT_SP1:
                position_specific_class = PositionSpecificLetter.SIGNAL_TAT_SP1
            else:
                position_specific_class = PositionSpecificLetter.CLEAVAGE_SITE_SP1

        if position_specific_class is None:
            print("Unexpected case", prev_letter, letter, next_letter)

        transformed = enc.transform([position_specific_class.value])
        classes_sequence.append(transformed)

    seq_tensor = np.array(classes_sequence).reshape((70))
    #  if sequence == "IIIIIIIIIIIMMMMMMMMMMMMMMMMMMMMMOOOOOMMMMMMMMMMMMMMMMMIIIIIIIIIIIIIIII":
    #  print("SE!", classes_sequence_to_letters(seq_tensor, enc))

    return seq_tensor


def classes_sequence_to_letters(seq_tensor, enc):
    letters = []
    for n in seq_tensor:
        letter = enc.inverse_transform([n])[0]
        letters.append(letter)

    return "".join(letters)
