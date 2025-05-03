import mujoco
import numpy as np

def compute_gravity_forces(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """
    Compute generalized gravity forces g(q) using mujoco.mj_rne.
    
    Args:
        model: mujoco.MjModel
        data: mujoco.MjData (qpos should be valid)
    
    Returns:
        g: Gravity torque vector of shape (nv,)
    """
    nv = model.nv

    # Backup state
    qpos_backup = data.qpos.copy()
    qvel_backup = data.qvel.copy()
    qacc_backup = data.qacc.copy()

    # Ensure valid qpos
    mujoco.mj_forward(model, data)

    # Set velocities and accelerations to zero
    data.qvel[:] = 0
    data.qacc[:] = 0

    # Allocate output
    g = np.zeros(nv)

    # Compute full inverse dynamics: gravity only when vel and acc = 0
    mujoco.mj_rne(model, data, 1, g)  # 1 = include gravity

    # Restore original state
    data.qpos[:] = qpos_backup
    data.qvel[:] = qvel_backup
    data.qacc[:] = qacc_backup
    mujoco.mj_forward(model, data)

    return g

def compute_coriolis_matrix(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """
    Compute the Coriolis + centrifugal matrix C(q, qdot) such that:
        c(q, qdot) = C(q, qdot) * qdot
    using finite differences and MuJoCo's mj_rne().

    Returns:
        C: Coriolis matrix of shape (nv, nv)
    """
    nv = model.nv
    C = np.zeros((nv, nv))

    # Backup state
    qpos_backup = data.qpos.copy()
    qvel_backup = data.qvel.copy()
    qacc_backup = data.qacc.copy()

    # Loop over unit vectors in velocity space
    for i in range(nv):
        # Reset velocities and accelerations
        data.qvel[:] = 0
        data.qacc[:] = 0
        mujoco.mj_forward(model, data)

        # Set unit velocity in i-th direction
        data.qvel[i] = 1.0

        # Compute c_i = C * e_i (i.e., i-th column of C)
        c_i = np.zeros(nv)
        mujoco.mj_rne(model, data, 0, c_i)  # exclude gravity (0)

        # Fill i-th column of C
        C[:, i] = c_i

    # Restore state
    data.qpos[:] = qpos_backup
    data.qvel[:] = qvel_backup
    data.qacc[:] = qacc_backup
    mujoco.mj_forward(model, data)

    return C