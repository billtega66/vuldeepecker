# verify_gpu_wsl.py
import tensorflow as tf

# optional: prevent TF from grabbing all VRAM
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPUs:", tf.config.list_physical_devices("GPU"))

# tiny GPU op
with tf.device("/GPU:0"):
    a = tf.random.uniform((1024, 1024))
    b = tf.random.uniform((1024, 1024))
    c = tf.matmul(a, b)
print("Matmul OK:", c.shape)
