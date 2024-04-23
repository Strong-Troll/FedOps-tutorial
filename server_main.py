import hydra
from omegaconf import DictConfig

from fedops.server.app import FLServer
import tensorflow_model
import tensorflow_data_preparation
from hydra.utils import instantiate



@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    
    """
    Set the initial global model you created in models.py.
    """
    # Build init global model using torch
    model = instantiate(cfg.model)
    model_type = cfg.model_type # Check tensorflow or torch model
    model_name = type(model).__name__
    gl_test_torch = tensorflow_model.test_tf() # set torch test    
    
    # Load validation data for evaluating global model
    x_train, x_test, x_val, y_train, y_test, y_val= tensorflow_data_preparation.load_partition_tf(dataset=cfg.dataset.name, 
                                                                        validation_split=cfg.dataset.validation_split, 
                                                                        batch_size=cfg.batch_size) 
    
    # Start fl server
    fl_server = FLServer(cfg=cfg, model=model, model_name=model_name, model_type=model_type,
                         x_val = x_val, y_val = y_val) # torch
    fl_server.start()
    

if __name__ == "__main__":
    main()

