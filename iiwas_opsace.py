import mujoco
import mujoco.viewer
import numpy as np
import time
import pinocchio as pino
from contact_estimator import high_gain_based_observer, kalman_disturbance_observer
from mujoco_dyn import compute_gravity_forces

# Cartesian impedance control gains.
impedance_pos = np.asarray([100.0, 100.0, 100.0])  # [N/m]
impedance_ori = np.asarray([50.0, 50.0, 50.0])  # [Nm/rad]

# Joint impedance control gains.
Kp_null = np.asarray([75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0])

# Damping ratio for both Cartesian and joint impedance control.
damping_ratio = 1.0

# Gains for the twist computation. These should be between 0 and 1. 0 means no
# movement, 1 means move the end-effector to the target in one integration step.
Kpos: float = 0.95

# Gain for the orientation component of the twist computation. This should be
# between 0 and 1. 0 means no movement, 1 means move the end-effector to the target
# orientation in one integration step.
Kori: float = 0.95

# Integration timestep in seconds.
integration_dt: float = 1.0

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002

def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # Load the model and data.
    model = mujoco.MjModel.from_xml_path("kuka_iiwa_14/scene.xml")
    data = mujoco.MjData(model)

    model.opt.timestep = dt
    
    # Load the model and data with Pinocchio
    pino_model = pino.buildModelFromMJCF("kuka_iiwa_14/iiwa14.xml")
    pino_data = pino_model.createData()

    # Compute damping and stiffness matrices.
    damping_pos = damping_ratio * 2 * np.sqrt(impedance_pos)
    damping_ori = damping_ratio * 2 * np.sqrt(impedance_ori)
    Kp = np.concatenate([impedance_pos, impedance_ori], axis=0)
    Kd = np.concatenate([damping_pos, damping_ori], axis=0)
    Kd_null = damping_ratio * 2 * np.sqrt(Kp_null)

    # End-effector site we wish to control.
    site_name = "attachment_site"
    site_id = model.site(site_name).id

    # Get the dof and actuator ids for the joints we wish to control. These are copied
    # from the XML file. Feel free to comment out some joints to see the effect on
    # the controller.
    joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    # Initial joint configuration saved as a keyframe in the XML file.
    key_name = "home"
    key_id = model.key(key_name).id
    q0 = model.key(key_name).qpos

    # Mocap body we will control with our mouse.
    mocap_name = "target"
    mocap_id = model.body(mocap_name).mocapid[0]

    # Pre-allocate numpy arrays.
    jac = np.zeros((6, model.nv))
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)
    M_inv = np.zeros((model.nv, model.nv))
    Mx = np.zeros((6, 6))

    # Initialize the contact estimator
    gm_estimator = high_gain_based_observer(dt, "hg", model.nv)
    # gm_estimator = kalman_disturbance_observer(dt, model.nv)

    # Initialize a buffer for visualization of the estimated external forces and generalized momentum.
    est_ext_wrenches = []
    est_gms = []
    gt_ext_wrenches = []
    gt_gms = []
    est_ext_taus = []
    gt_ext_taus = []

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        # Reset the simulation.
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Enable site frame visualization.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        while viewer.is_running():
            step_start = time.time()

            # Spatial velocity (aka twist).
            dx = data.mocap_pos[mocap_id] - data.site(site_id).xpos
            twist[:3] = Kpos * dx / integration_dt
            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
            mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
            twist[3:] *= Kori / integration_dt

            # Jacobian.
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

            # Compute the task-space inertia matrix.
            mujoco.mj_solveM(model, data, M_inv, np.eye(model.nv))
            Mx_inv = jac @ M_inv @ jac.T
            if abs(np.linalg.det(Mx_inv)) >= 1e-2:
                Mx = np.linalg.inv(Mx_inv)
            else:
                Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)

            # Compute the dense mass matrix using MuJoCo
            M = np.zeros((model.nv, model.nv))  # Pre-allocate the dense matrix
            mujoco.mj_forward(model, data) # Warning: don't forget to call mj_forward before mj_fullM
                                           # to update the dynamics state                      
            mujoco.mj_fullM(model, M, data.qM)
            
            # Compute the Coriolis matrix with pinocchio
            C_pino = pino.computeCoriolisMatrix(pino_model, pino_data, data.qpos, data.qvel) 
            # Compute the Gravitry vector with MuJoCo
            g = compute_gravity_forces(model, data)
            
            # Compute generalized forces.
            tau = jac.T @ Mx @ (Kp * twist - Kd * (jac @ data.qvel[dof_ids]))

            # Add joint task in nullspace.
            Jbar = M_inv @ jac.T @ Mx
            ddq = Kp_null * (q0 - data.qpos[dof_ids]) - Kd_null * data.qvel[dof_ids]
            tau += (np.eye(model.nv) - jac.T @ Jbar.T) @ ddq

            # Add gravity compensation.
            if gravity_compensation:
                tau += data.qfrc_bias[dof_ids]
            
            # Update the generalized momentum observer
            gt_gm = M @ data.qvel
            est_ext_tau, est_gm = gm_estimator.update(data.qvel, M, C_pino, g, tau)
            
            # Print out the external forces on the end-effector.
            ee_body_name = "attachment"
            ee_body_id = model.body(ee_body_name).id
            ext_wrench = data.xfrc_applied[ee_body_id]

            # Compute the body jacobian for the end-effector.
            jac_body = np.zeros((6, model.nv))
            mujoco.mj_jacBody(model, data, jac_body[:3], jac_body[3:], ee_body_id)
            est_ext_wrench = np.linalg.pinv(jac_body.T) @ est_ext_tau

            # Append the estimated external forces and generalized momentum to the buffer.
            est_ext_wrenches.append(est_ext_wrench.copy())
            est_gms.append(est_gm.copy())
            gt_ext_wrenches.append(ext_wrench.copy())
            gt_gms.append(gt_gm.copy())

            # Set the control signal and step the simulation.
            np.clip(tau, *model.actuator_ctrlrange.T, out=tau)
            data.ctrl[actuator_ids] = tau[actuator_ids]
            mujoco.mj_step(model, data)
            
            # Compute the inverse dynamics torques.
            mujoco.mj_forward(model, data)
            mujoco.mj_inverse(model, data)

            # The total joint torques (including gravity, Coriolis, and external forces)
            tau_total = data.qfrc_inverse.copy()
            
            # Compute the joint space external torques
            gt_ext_tau = tau_total - tau
            gt_ext_taus.append(gt_ext_tau.copy())
            est_ext_taus.append(est_ext_tau.copy())

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # Visualize the estimated external forces and generalized momentum.
    est_ext_wrenches = np.array(est_ext_wrenches)
    est_gms = np.array(est_gms)
    est_ext_taus = np.array(est_ext_taus)
    gt_ext_wrenches = np.array(gt_ext_wrenches)
    gt_gms = np.array(gt_gms)
    gt_ext_taus = np.array(gt_ext_taus)
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 5))
    plot_param = 611
    axs_name = ["fx", "fy", "fz", "mx", "my", "mz"]
    t = np.arange(len(est_ext_wrenches)) * dt
    for i in range(6):
        ax = fig.add_subplot(plot_param)
        ax.plot(t, est_ext_wrenches[:, i], label=f"Estimated External {axs_name[i]}")
        ax.plot(t, gt_ext_wrenches[:, i], label=f"Ground Truth External {axs_name[i]}")
        ax.legend()
        plot_param += 1

    fig = plt.figure(figsize=(10, 5))
    plot_param = 711
    axs_name = ["gm1", "gm2", "gm3", "gm4", "gm5", "gm6", "gm7"]
    for i in range(7):
        ax = fig.add_subplot(plot_param)
        ax.plot(t, est_gms[:, i], label=f"Estimated Generalized Momentum {axs_name[i]}")
        ax.plot(t, gt_gms[:, i], label=f"Ground Truth Generalized Momentum {axs_name[i]}")
        ax.legend()
        plot_param += 1
    
    fig = plt.figure(figsize=(10, 5))
    plot_param = 711
    axs_name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
    for i in range(7):
        ax = fig.add_subplot(plot_param)
        ax.plot(t, est_ext_taus[:, i], label=f"Estimated External Torque {axs_name[i]}")
        ax.plot(t, gt_ext_taus[:, i], label=f"Ground Truth External Torque {axs_name[i]}")
        ax.legend()
        plot_param += 1

    plt.show()


if __name__ == "__main__":
    main()