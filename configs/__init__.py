from pathlib import Path

CONFIG_ROOT = Path(__file__).parent

class ConfigManager:
    @staticmethod
    def get_model_config(model_name="yolov10"):
        return CONFIG_ROOT / "model" / f"{model_name}.yaml"
    
    # ... 其他配置获取方法 ...