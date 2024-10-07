# import hydra
# from omegaconf import DictConfig

# from fedops.server.app import FLServer
# import tensorflow_model
# import tensorflow_data_preparation
# from hydra.utils import instantiate



# @hydra.main(config_path="conf", config_name="config", version_base=None)
# def main(cfg: DictConfig) -> None:
    
#     """
#     Set the initial global model you created in models.py.
#     """
#     # Build init global model using torch
#     model = instantiate(cfg.model)
#     model_type = cfg.model_type # Check tensorflow or torch model
#     model_name = type(model).__name__
#     gl_test_torch = tensorflow_model.test_tf() # set torch test    
    
#     # Load validation data for evaluating global model
#     x_train, x_test, x_val, y_train, y_test, y_val= tensorflow_data_preparation.load_partition_tf(dataset=cfg.dataset.name, 
#                                                                         validation_split=cfg.dataset.validation_split, 
#                                                                         batch_size=cfg.batch_size) 
#     model = model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     # model.build((None, 28, 28, 1))  # Build the model if not already built
#     # Start fl server
#     fl_server = FLServer(cfg=cfg, model=model, model_name=model_name, model_type=model_type,
#                          x_val = x_val, y_val = y_val) # torch
#     fl_server.start()
    

# if __name__ == "__main__":
#     main()

import hydra 
from omegaconf import DictConfig

# Assuming these imports are correctly defined in your project
from fedops.server.app import FLServer
import tensorflow_model
import tensorflow_data_preparation
from hydra.utils import instantiate


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    
    """
    Set the initial global model you created in models.py.
    """
    # Instantiate the model as per configuration
    model = tensorflow_model.MNISTClassifier()
    print("////////////////////////////////////////////////////////////////////////", model.get_weights())
    model_type = cfg.model_type  # Determine if the model is TensorFlow or Torch

    # Correct function based on model_type
    if model_type == 'Tensorflow':
        # test_function = tensorflow_model.test_tf  # Adjusted for TensorFlow testing function
        # Compile the TensorFlow model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # No need to build explicitly if using `compile` and `fit` methods, but uncomment if necessary
        model.build((None, 28, 28, 1))  # Only if you're sure about the input shape
        print("Model build Success!!!~~~!!!~~~!!!~~~!!!~~~!!!~~~!!!~~~!!!~~~!!!~~~!!!")

    # Load validation data for evaluating the global model
    x_train, x_test, x_val, y_train, y_test, y_val = tensorflow_data_preparation.load_partition_tf(
        dataset=cfg.dataset.name,
        validation_split=cfg.dataset.validation_split,
        batch_size=cfg.batch_size)

    print("////////////////////////////////////////////////////////////////////////", model.get_weights())
    # Initialize FL server with the appropriate model and data
    fl_server = FLServer(cfg=cfg, model=model, model_name=model.__class__.__name__, model_type=model_type,
                         x_val=x_val, y_val=y_val)
    fl_server.start()
    

if __name__ == "__main__":
    main()

