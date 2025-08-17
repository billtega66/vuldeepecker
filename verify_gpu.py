# at the top of verify_gpu.py
import os, sys

if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin")


import tensorflow as tf
print("GPUs:", tf.config.list_physical_devices('GPU'))

# optional: detailed build info
print(tf.sysconfig.get_build_info())
