"""Make the crystal-TI scripts importable from these tests.

The scripts in the parent directory are standalone command-line tools, not an installed package,
so the tests put that directory on ``sys.path`` themselves. This lets ``pytest`` be run from the
repository root (which collects these tests along with ``colloids/tests``) as well as from here.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
