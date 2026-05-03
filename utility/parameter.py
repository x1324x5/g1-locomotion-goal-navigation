import yaml
import os
import torch


class Parameter:
    def __init__(self, config_path, arg_cli=None):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        setattr(self, "config_path", config_path)
        with open(config_path, "r") as file:
            self._config_dict = yaml.safe_load(file)
        if arg_cli:
            self._merge_cli_args(arg_cli)

        for key, value in self._config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, DictConfig(value))
            else:
                setattr(self, key, value)

        if arg_cli:
            for k, v in vars(arg_cli).items():
                if not hasattr(self, k):
                    setattr(self, k, v)

    def _merge_cli_args(self, arg_cli):
        cli_args_dict = {k: v for k, v in vars(arg_cli).items() if v is not None}
        for k, v in cli_args_dict.items():
            updated = self.update(self._config_dict, k, v)
            if not updated:
                self._config_dict[k] = v
        pass

    def update(self, config_dict, target_key, target_v):
        if target_key in config_dict:
            assert not isinstance(
                config_dict[target_key], dict
            ), "Cannot update a dictionary value directly."
            config_dict[target_key] = target_v
            return True
        for k, v in config_dict.items():
            if isinstance(v, dict):
                if self.update(v, target_key, target_v):
                    return True
        return False


class DictConfig:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DictConfig(value))
            else:
                setattr(self, key, value)

    def __repr__(self):
        return str(self.__dict__)
