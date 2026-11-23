import math
import struct
from unittest.mock import patch

import pytest

from examples.classified_qwen import Session


def test_replay_requires_byte_equality_to_last_success():
    """Cache(F,x)=F(x); x′≠x incurs a new evaluation even if |x′−x|=1 ulp."""
    x = struct.pack("<d", 1.)
    adjacent = struct.pack("<d", math.nextafter(1., 2.))
    session = object.__new__(Session)
    session.directory = None
    session.manifest = {"hidden": 1}
    session.last = None
    with patch("examples.classified_qwen.advance", return_value=([1.], 1)) as evaluate:
        mutable = bytearray(x)
        assert session.dispatch(mutable)[0] == "interpret"
        mutable[:] = adjacent
        assert session.dispatch(x)[0] == "replay"
        assert session.dispatch(mutable)[0] == "interpret"
        assert session.dispatch(x)[0] == "interpret"
        assert evaluate.call_count == 3
        with pytest.raises(ValueError, match="finite"):
            session.dispatch(struct.pack("<d", math.nan))
        assert session.dispatch(x)[0] == "replay"
