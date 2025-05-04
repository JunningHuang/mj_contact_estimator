import mujoco
import pinocchio as pino
from pathlib import Path
import numpy as np

def get_foot_transform(foot_name):
    id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{foot_name}")
    
    # compute the transformation from world to the foot frame
    foot_pos_world = data.geom_xpos[id]
    foot_rot_world = data.geom_xmat[id].reshape(3, 3).T # from local to world
    
    return foot_pos_world, foot_rot_world

# TODO: add a new frame of the foot 
def add_frame_to_model(pino_model, new_frame_name, parent_id, previousFrame, foot_name):
    foot_pos_world, foot_rot_world = get_foot_transform(foot_name)
    se3_pose = pino.SE3(foot_rot_world, foot_pos_world)
    new_frame = pino.Frame(new_frame_name, parent_id, previousFrame, se3_pose, pino.FrameType.OP_FRAME)
    pino_model.addFrame(new_frame)
    return new_frame

# TODO: add foot frames to the pinocchio model
def add_frames_to_model(pino_model, foot_names, body_names, joint_names):
    """
    Add foot frames to the pinocchio model
    :param pino_model: pinocchio model
    :param foot_names: list of foot names
    :param body_names: list of body names that correspond to the foot names, usually the calf
    """
    frames = {}
    for i in range(len(foot_names)):
        foot_name = foot_names[i]
        body_name = body_names[i]
        joint_name = joint_names[i]
        previousFrameid = pino_model.getFrameId(body_name)
        parentJointId = pino_model.getJointId(joint_name)
        new_frame_name = f"{foot_name}"
        new_frame = add_frame_to_model(pino_model, new_frame_name, parentJointId, previousFrameid, foot_name)
        frames[foot_name] = new_frame
        previousFrame = new_frame
    return frames

def gen_motion_mujoco2pino(model, data):
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    
    gen_vel_local = np.empty(6)
    local_coord_flag = True
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, trunk_id, gen_vel_local, local_coord_flag)
    ang_vel_local = gen_vel_local[:3]
    lin_vel_local = gen_vel_local[3:]
    
    gen_acc_local = np.empty(6)
    mujoco.mj_objectAcceleration(model, data, mujoco.mjtObj.mjOBJ_XBODY, trunk_id, gen_acc_local, local_coord_flag)
    ang_acc_local = gen_acc_local[:3]
    lin_acc_local = gen_acc_local[3:]
    
    pino_q = data.qpos
    pino_trunk_pos = pino_q[: 3]
    mujoco_trunk_quat = pino_q[3: 7]
    pino_trunk_quat = [mujoco_trunk_quat[1], mujoco_trunk_quat[2], mujoco_trunk_quat[3], mujoco_trunk_quat[0]]
    pino_q = np.hstack([pino_trunk_pos, pino_trunk_quat, data.qpos[7:]]) # convert trunk pose from mujoco to pinocchio
    
    pino_v = np.hstack([lin_vel_local, ang_vel_local, data.qvel[6:]])
    pino_a = np.hstack([lin_acc_local, ang_acc_local, data.qacc[6:]])
    return pino_q, pino_v, pino_a 

if __name__ == "__main__":    
    xml_path = (Path(__file__).resolve().parent / "data" / "plane.xml").as_posix()
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # load the pinocchio model from the urdf
    urdf_path = (Path(__file__).resolve().parent / "data" / "a1.urdf").as_posix()
    pino_mujoco_model = pino.buildModelFromMJCF(xml_path)
    pino_mujoco_data = pino.Data(pino_mujoco_model)
        
    # verify the dynamic parameters and kinematics of the two models
    pino_q, pino_v, pino_a = gen_motion_mujoco2pino(model, data) # convert motion data from mujoco to pinocchio
    
    # compare the jacobian of mujoco and pinocchio output
    body_name = "FR_calf"
    id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{body_name}")
    J_lin = np.zeros((3, model.nv))
    J_rot = np.zeros((3, model.nv))
    
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_jacBody(model, data, J_lin, J_rot, id)
    J = np.vstack([J_lin, J_rot])
    
    # 
    pino_frame_id = pino_mujoco_model.getFrameId(body_name)
    pino.forwardKinematics(pino_mujoco_model, pino_mujoco_data, pino_q)
    pino.computeJointJacobians(pino_mujoco_model, pino_mujoco_data)
    pino.updateFramePlacement(pino_mujoco_model, pino_mujoco_data, pino_frame_id)
    J_ = pino.getFrameJacobian(pino_mujoco_model, pino_mujoco_data, pino_frame_id, pino.ReferenceFrame.LOCAL_WORLD_ALIGNED)

    add_frames_to_model(pino_mujoco_model, ["FR_foot", "FL_foot", "RR_foot", "RL_foot"], ["FR_calf", "FL_calf", "RR_calf", "RL_calf"], ["FR_calf_joint", "FL_calf_joint", "RR_calf_joint", "RL_calf_joint"])
    