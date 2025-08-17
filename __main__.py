from representation.vectorize import vectorize_and_save
from model.train_blstm import train
from model.evaluate_model import evaluate
import tensorflow as tf
# If the project uses tf.compat.v1 APIs, keep TF2 but run in v1-compat mode:
#tf.compat.v1.disable_eager_execution()
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()
# Prevent TF from grabbing all VRAM
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == "__main__":
    print("[1/3] Vectorizing gadgets...")
    vectorize_and_save()

    print("[2/3] Training BLSTM model...")
    train()

    print("[3/3] Evaluating trained model...")
    evaluate()