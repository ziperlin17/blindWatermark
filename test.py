import os
cuda_bin_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin"
if os.path.exists(cuda_bin_path):
    try:
        os.add_dll_directory(cuda_bin_path)
        print(f"Added to DLL search path: {cuda_bin_path}")
    except AttributeError:
        print("os.add_dll_directory not available (Python < 3.8 or non-Windows).")
    except Exception as e:
        print(f"Error adding DLL directory: {e}")
else:
    print(f"CUDA bin path not found: {cuda_bin_path}")

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))
# Попробуйте выполнить простую операцию
try:
    with tf.device('/GPU:0'):
         a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
         b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
         c = tf.matmul(a, b)
    print("GPU operation successful:", c)
except RuntimeError as e:
    print("GPU operation failed:", e)
except Exception as e:
    print("Error:", e)