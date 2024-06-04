import tensorflow as tf

# Check for GPU availability
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("GPU is available and configured.")
    except RuntimeError as e:
        print(e)
else:
    print("GPU is not available, using CPU.")
