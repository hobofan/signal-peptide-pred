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
