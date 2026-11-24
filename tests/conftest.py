import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from core.workspace import Workspace


@pytest.fixture
def workspace(tmp_path) -> Workspace:
    """A workspace rooted at a temporary directory — tests never write into the real output/."""
    return Workspace(tmp_path)
