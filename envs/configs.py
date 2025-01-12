from typing import NamedTuple, Dict, Any

class EnvConfig:
    """
    Assuming static configs.
    """
    kinematics_config = {
        "observation": {
            "normalize": False,
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        },
        "duration": 50,
        "vehicles_count": 20,
        "vehicles_density": 2,
    }

    occupancy_grid_config = {
        "vehicles_count": 10,
        "observation": {
            "normalize": False,
            "vehicles_count": 5,
            "type": "OccupancyGrid",
            "features": ["x", "y", "vx", "vy"],
            "features_range": {
                "x": [-50, 50],
                "y": [-50, 50],
                "vx": [-10, 10],
                "vy": [-10, 10]
            },
            "grid_size": [[-20, 20], [-20, 20]],
            "grid_step": [5, 5],
            "absolute": False
        }
    }
    time_to_collision = {"observation": {"normalize": False, "type": "TimeToCollision", "horizon": 10}}

    @staticmethod
    def get_config(obs_type: str) -> Dict[str, Any]:
        configs = {
            "Kinematics": EnvConfig.kinematics_config,
            "OccupancyGrid": EnvConfig.occupancy_grid_config,
            "TimeToCollision": EnvConfig.time_to_collision}
        return configs.get(obs_type)
