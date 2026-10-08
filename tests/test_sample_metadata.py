import threading

import numpy as np

from oculus_reader.reader import OculusReader


def test_snapshot_exposes_one_sequence_for_one_acquired_sample():
    reader = OculusReader.__new__(OculusReader)
    reader._lock = threading.Lock()
    reader.last_transforms = {"l": np.eye(4)}
    reader.last_buttons = {"X": False}
    reader.last_sample_sequence = 7
    reader.last_sample_monotonic = 12.5

    transforms, buttons, sequence, acquired = reader.get_snapshot()
    assert transforms is reader.last_transforms
    assert buttons is reader.last_buttons
    assert sequence == 7
    assert acquired == 12.5

    # Polling does not manufacture a new source sample.
    assert reader.get_snapshot()[2] == sequence
