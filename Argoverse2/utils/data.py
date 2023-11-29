import os 
import numpy as np

def pad_obj_vectors(obj_vec, ts, n_vec=60, inf=1000):
    """
    Pad object vectors with first frame at the beginning and last frame at the end to match the desired number of vectors.

    Args:
        obj_vec (np.ndarray): Object vectors to be padded.
        ts (np.ndarray): Timestamps associated with the object vectors.
        n_vec (int): Desired number of vectors.
        inf (int): Padding value.

    Returns:
        np.ndarray: Padded object vectors.
    """
    diff_s = int(ts[0])
    diff_e = int(n_vec - ts[-1])

    obj_vec_tensor = np.array(obj_vec)
    # [xs, ys, xe, ye, ts_vg, dis, ang, vx, vy, heading, obj_type, Pid]

    pad_s = [obj_vec[0].tolist()]
    pad_e = [obj_vec[-1].tolist()]
    pad_s = np.array(pad_s * diff_s)
    pad_e = np.array(pad_e * diff_e)

    if len(pad_s) == 0:
        pad_s = np.empty((0, obj_vec.shape[1]))

    if len(pad_e) == 0:
        pad_e = np.empty((0, obj_vec.shape[1]))

    obj_vec = np.vstack([pad_s, obj_vec_tensor, pad_e])
    return obj_vec



def pad_lane_vectors(lane_vec, n_vec=35, inf=1000):
    """
    Pad lane vectors with zeros at the beginning to match the desired number of vectors.

    Args:
        lane_vec (np.ndarray): Lane vectors to be padded.
        n_vec (int): Desired number of vectors.
        inf (int): Padding value.

    Returns:
        np.ndarray: Padded lane vectors.
    """
    diff = n_vec - len(lane_vec)

    lane_vec_tensor = np.array(lane_vec)
    # [xs, ys, zs, xe, ye, ze, is_inter, dir, type, line_id]

    pad = [lane_vec_tensor[0].tolist()]
    pad = np.array(pad * diff)
    
    if len(pad) > 0:
        lane_vec = np.vstack([pad, lane_vec_tensor])
    
    return lane_vec[:n_vec]


def save_agent(agent_vector, pref, main_dir, agent_dir):
    """
    Save agent vectors to a file.

    Args:
        agent_vector (np.ndarray): Agent vectors to be saved.
        pref (str): Prefix for the file name.
        main_dir (str): Main directory for saving.
        agent_dir (str): Subdirectory for agent data.

    Returns:
        None
    """
    file_name = f"{pref}_agent_vector"
    file_path = os.path.join(main_dir, agent_dir, file_name)

    np.save(file_path, agent_vector)



def save_objects(vectors, pref, main_dir, obj_dir, n_obj):
    """
    Save object vectors to separate files.

    Args:
        vectors (np.ndarray): Object vectors to be saved.
        pref (str): Prefix for the file names.
        main_dir (str): Main directory for saving.
        obj_dir (str): Subdirectory for object data.

    Returns:
        None
    """
    file_name = pref + "_obj_vector"
    file_path = os.path.join(main_dir, obj_dir, file_name)
    data = {
        'vec' : vectors,
        'n_obj' : n_obj
    }
    np.savez(file_path, **data)


def save_lanes(vectors, pref, main_dir, lane_dir, n_lane):
    """
    Save lane vectors to separate files.

    Args:
        vectors (np.ndarray): Lane vectors to be saved.
        pref (str): Prefix for the file names.
        main_dir (str): Main directory for saving.
        lane_dir (str): Subdirectory for lane data.

    Returns:
        None
    """
    file_name = pref + "_lane_vector"
    file_path = os.path.join(main_dir, lane_dir, file_name)
    data = {
        'vec' : vectors,
        'n_lane' : n_lane
    }
    np.savez(file_path, **data)


def save_gt(gt, pref, main_dir, gt_dir):
    """
    Save ground truth vectors to a file.

    Args:
        gt (np.ndarray): Ground truth vectors to be saved.
        pref (str): Prefix for the file name.
        main_dir (str): Main directory for saving.
        gt_dir (str): Subdirectory for ground truth data.

    Returns:
        None
    """
    file_name = f"{pref}_gt"
    file_path = os.path.join(main_dir, gt_dir, file_name)
    np.save(file_path, gt)
