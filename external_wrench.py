import mujoco
import numpy as np

def apply_external_wrench(model: mujoco.MjModel, data: mujoco.MjData, wrench: np.ndarray, body_name: str) -> None:
    """
    Apply external wrenches to the specified joints in the MuJoCo model.

    Args:
        model: mujoco.MjModel
        data: mujoco.MjData
        wrench: Wrench vector to be applied (6D)
        body_name: Body name to which the wrenches will be applied
    """
    joint_id = model.body(body_name).id
    data.xfrc_applied[joint_id] = wrench

def generate_wrench_profile(t: list, wrench_type: str) -> np.ndarray:
    """
    Generate a wrench profile based on the specified type and time.

    Args:
        t: A list of time sequence (in seconds) (N, )
        wrench_type: Type of wrench profile ('step', 'sine')

    Returns:
        A list of wrench vector (N, 6D)
    """
    N = len(t)
    wrenches = np.zeros((N, 6))
    if wrench_type == "step":
        wrenches[:, 0] = 0.0
        wrenches[:, 1] = 0.0
        wrenches[:, 2] = 10.0
    elif wrench_type == "sine":
        wrenches[:, 0] = 0.0 
        wrenches[:, 1] = 0.0
        wrenches[:, 2] = 10.0 * np.sin(2 * np.pi * t)
    else:   
        raise ValueError("Invalid wrench type. Choose 'step' or 'sine'.")
    return wrenches

class WrenchApplier:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, wrench_type: str, time_stop: float = 0.0, dt: float = 0.01, body_names: list = None):
        self.model = model
        self.data = data
        self.wrench_type = wrench_type
        self.wrench_profile = None
        self.wrench_profile_index = 0
        self.time_stop = time_stop
        self.dt = dt
        self.body_names = body_names if body_names is not None else []
        self._init_wrench_profile()

    def _init_wrench_profile(self):
        """
        Initialize the wrench profile for the specified body names and time sequence.

        Args:
            body_names: List of body names to which the wrenches will be applied
            t: A list of time sequence (in seconds) (N, )
        """
        t = np.arange(0, self.time_stop+self.dt, self.dt)
        wrench_profile = generate_wrench_profile(t, self.wrench_type)
        self.t = t
        self.wrench_profile = {body_name: wrench_profile for body_name in self.body_names}
    
    def apply_wrench(self):
        """
        Apply the wrench profile to the specified body names.
        """
        if self.wrench_profile is None:
            raise ValueError("Wrench profile not initialized. Call _init_wrench_profile() first.")
        
        applied_external_wrench = {}
        if self.wrench_profile_index < len(self.t):
            for body_name, wrench in self.wrench_profile.items():
                apply_external_wrench(self.model, self.data, wrench[self.wrench_profile_index], body_name)
                applied_external_wrench[body_name] = wrench[self.wrench_profile_index]
                
        self.wrench_profile_index += 1
        return applied_external_wrench
        
    def reset(self):
        """
        Reset the wrench profile index to 0.
        """
        self.wrench_profile_index = 0