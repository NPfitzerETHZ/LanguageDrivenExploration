from omegaconf import DictConfig, OmegaConf
import hydra
import sys

# def extract_initial_config():
#     config_name, restore_path = None, None
#     for arg in sys.argv:
#         if arg.startswith("cfg_name="):
#             config_name = arg.split("=", 1)[1]
#         elif arg.startswith("restore_path="):
#             restore_path = arg.split("=", 1)[1]
#     return config_name, restore_path

# cfg_name, restore_path = extract_initial_config()
# print(f"Config name: {cfg_name}")
@hydra.main(version_base=None,config_path="../../conf")
def main(cfg: DictConfig) -> None:
    
    print(OmegaConf.to_yaml(cfg, resolve=True))   # full merged config
    
if __name__ == '__main__':
    main()