"Testing of pointnet module."
import sys
import pytest

def test_dryrun():
  sys.argv = ["train.py", "--dryrun"]
  import train  # pylint: disable=import-outside-toplevel, unused-import

if __name__ == "__main__":
  sys.exit(pytest.main([__file__]))
