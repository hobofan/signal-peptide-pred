from enum import Enum


class Kingdom(Enum):
    EUKARYA = "EUKARYA"
    ARCHAEA = "ARCHAEA"
    # Gram-negative
    NEGATIVE = "NEGATIVE"
    # Gram-positive
    POSITIVE = "POSITIVE"

    @classmethod
    def values(cls):
        return [e.value for e in cls]


class SPType(Enum):
    # No signal peptide
    NO_SP = "NO_SP"
    # Sec/SP1
    SP = "SP"
    # Tat/SP1
    TAT = "TAT"
    # Sec/SP2
    LIPO = "LIPO"

    @classmethod
    def values(cls):
        return [e.value for e in cls]


class AnnotationLetter(Enum):
    INNER = "I"
    OUTER = "O"
    TRANSMEMBRANE = "M"
    # Includes cleavage site AA
    SIGNAL_SEC_SP1 = "S"
    # Includes cleavage site AA
    SIGNAL_SEC_SP2 = "L"
    # Includes cleavage site AA
    SIGNAL_TAT_SP1 = "T"


class PositionSpecificLetter(Enum):
    """Position-specfic annotation of a letter (= amino acid).

    In addition to the general letter annotation, it also adds position-specific
    information based on the neighboring letters, e.g. whether a transmembrane amino acid
    is entering or leaving the membrane.
    """

    SIGNAL_SEC_SP1 = "S"
    SIGNAL_SEC_SP2 = "Z"
    SIGNAL_TAT_SP1 = "T"
    CLEAVAGE_SITE_SP1 = "C"
    CLEAVAGE_SITE_SP2 = "K"
    OUTER = "O"
    INNER = "I"
    TRANSMEMBRANE_IN_OUT = "L"
    TRANSMEMBRANE_OUT_IN = "E"

    @classmethod
    def values(cls):
        return [e.value for e in cls]
