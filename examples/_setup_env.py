
import sys
import os


my_dir = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
root_dir = os.path.dirname(my_dir)

if root_dir not in sys.path:
  sys.path.insert(0, root_dir)
