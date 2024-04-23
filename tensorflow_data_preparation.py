import tensorflow as tf
import json
import logging
from datetime import datetime

# Set up logging
handlers_list = [logging.StreamHandler()]
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=handlers_list)

logger = logging.getLogger(__name__)

# TensorFlow version of MNIST data loader
def load_partition_tf(dataset, validation_split, batch_size):
    """
    The variables train_dataset, val_dataset, and test_dataset must be returned fixedly.
    """
    now = datetime.now()
    now_str = now.strftime('%Y-%m-%d %H:%M:%S')
    fl_task = {"dataset": "MNIST", "start_execution_time": now_str}
    fl_task_json = json.dumps(fl_task)
    logging.info(f'FL_Task - {fl_task_json}')

    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize and reshape data
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    # Create validation split from training data
    num_validation_samples = int(validation_split * x_train.shape[0])
    x_val = x_train[:num_validation_samples]
    y_val = y_train[:num_validation_samples]
    x_train = x_train[num_validation_samples:]
    y_train = y_train[num_validation_samples:]

    # Create tf.data datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    return x_train, x_test, x_val, y_train, y_test, y_val

def gl_model_tf_validation(batch_size):
    """
    Setting up a dataset to evaluate a global model on the server
    """
    # Load the test set of MNIST Dataset
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test / 255.0
    x_test = x_test[..., tf.newaxis]

    # Create tf.data dataset for validation
    gl_val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    return gl_val_dataset
