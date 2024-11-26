import yaml
from typing import Any, Dict, Optional


def load_config(
    config: Optional[Dict[str, Any]] = None, 
    config_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load configuration from a dictionary or a YAML file.
    
    This function allows flexibility in loading configuration parameters either directly
    from a provided dictionary or by reading from a specified YAML file. It ensures that
    the necessary configuration is available for initializing various components of the
    BioNeuralNet package.
    
    Args:
        config (dict, optional): 
            Configuration dictionary containing key-value pairs for settings.
            If provided, this dictionary will be used directly.
        config_path (str, optional): 
            Path to the configuration YAML file. 
            This is used only if `config` is not provided.
    
    Returns:
        dict: 
            Loaded configuration dictionary containing all necessary parameters.
    
    Raises:
        ValueError: 
            If neither `config` nor `config_path` is provided.
        FileNotFoundError: 
            If the specified `config_path` does not exist.
        yaml.YAMLError: 
            If there is an error parsing the YAML configuration file.
    """
    if config is None and config_path is None:
        raise ValueError("Either a configuration dictionary or a configuration file path must be provided.")
    
    if config is None:
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"The configuration file was not found: {config_path}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing the YAML configuration file: {e}")
    
    return config
