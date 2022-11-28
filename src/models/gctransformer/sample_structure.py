from dataclasses import dataclass


@dataclass
class SampleStructure:
    pad_token: int
    start_token: int
    end_token: int
    token_shift: int


def get_sample_structure(version: int) -> SampleStructure:
    if version == 0:
        return SampleStructure(pad_token=0, start_token=1, end_token=2, token_shift=3)

    if version == 1:
        return SampleStructure(pad_token=-1, start_token=0, end_token=1, token_shift=2)

    if version == 2:
        return SampleStructure(pad_token=-1, start_token=0, end_token=-1, token_shift=1)

    raise ValueError(f"unknown sample structure {version}")
