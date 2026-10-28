import sys
from pathlib import Path

# the RL modules live one directory up (flat layout, imported by name as in the run scripts)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
