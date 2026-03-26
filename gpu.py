import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# List GPUs
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)

if gpus:
    print("✅ GPU is detected by TensorFlow")

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Show detailed GPU info
    print(tf.config.experimental.get_device_details(gpus[0]))
else:
    print("❌ No GPU detected by TensorFlow")