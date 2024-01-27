import numpy as np 
from config import * 
import tensorflow as tf



def extract_agent_features(i, xy, velocity, yaw, gt_xy, is_avail, DISP) :

  # timesteps
  ts = np.arange(len(xy))

  # Candidate Density
  agent_disp = DISP[i]

  # average velocity
  total_vel = np.sqrt(velocity[:, 0]**2 + velocity[:, 1]**2)
  avg_vel = np.mean(total_vel)

  radius = OFFSET * avg_vel
  CD = []
  for j in range(11):
    disp = DISP[i, :, j] <= radius
    indices = np.where(disp)
    cd = len(indices[0])
    CD.append(cd)

  # Normalized_GT
  center = xy[-1]
  Normalized_GT = gt_xy-center

  agent_data = {
      'Agent_XY' : xy,
      'Agent_ts' : ts,
      'Agents_v' : velocity,
      'Agent_H' : yaw,
      'CD' : np.array(CD),
      'GT' : gt_xy,
      'Normalized_GT' : Normalized_GT,
      'IS_AVAIL' : is_avail
  }

  return radius, agent_data





def extract_object_features(xy, obj_xy, obj_velocity, obj_yaw, Agent_type):

  # distance from agent
  dis = np.linalg.norm(xy - obj_xy, axis=1)

  # angle to agent
  theta = np.arctan2(obj_xy[:, 1]-xy[:, 1], obj_xy[:, 0]-xy[:, 0])

  obj_data = {
      'Object_pastXY' : obj_xy,
      'Distance_from_Agent' : dis,
      'Angle_to_Agent' : theta,
      'Object_Heading' : obj_yaw,
      'Object_velocity' : obj_velocity,
      'Object_type' : Agent_type
  }

  return obj_data




def vectorize_agent(agent_data):
  xy = agent_data['Agent_XY']
  xys = xy[:-1]
  xye = xy[1:]

  cd = agent_data['CD']
  cdv = np.vstack([cd[:-1], cd[1:]])
  cd = cdv.mean(axis=0)

  h = agent_data['Agent_H']
  hv = np.vstack([h[:-1], h[1:]])
  h = hv.mean(axis=0)

  vxy = np.expand_dims(agent_data['Agents_v'], -1)
  vxy = np.concatenate([vxy[:-1], vxy[1:]], -1)
  vxy = vxy.mean(axis=-1)


  vectors = np.vstack([xys[:, 0], xys[:, 1], xye[:, 0], xye[:, 1], cd, vxy[:, 0], vxy[:, 1], h]).T
  return vectors





def vectorize_object(_obj_data):
  vectors = np.empty((0, 11))
  for i in range(len(_obj_data)):
    xys = _obj_data[i]['Object_pastXY'][:-1]
    xye = _obj_data[i]['Object_pastXY'][1:]

    dis = np.vstack([_obj_data[i]['Distance_from_Agent'][:-1], _obj_data[i]['Distance_from_Agent'][1:]]).mean(axis=0)

    angle = np.vstack([_obj_data[i]['Angle_to_Agent'][:-1], _obj_data[i]['Angle_to_Agent'][1:]]).mean(axis=0)

    heading = np.vstack([_obj_data[i]['Object_Heading'][:-1], _obj_data[i]['Object_Heading'][1:]]).mean(axis=0)

    vxy = np.expand_dims(_obj_data[i]['Object_velocity'], -1)
    vxy = np.concatenate([vxy[:-1], vxy[1:]], axis=-1).mean(axis=-1)

    ts = np.arange(0, len(vxy))

    typ = np.ones(len(xys))*(_obj_data[i]['Object_type'])

    vector = np.vstack([xys[:, 0], xys[:, 1], xye[:, 0], xye[:, 1], ts, dis, angle, vxy[:,0], vxy[:, 1], heading, typ]).T
    if len(vector)<10:
      n = 10 - len(vector)
      to_pad = np.array([vector[-1] for j in range(n)])
      vector = np.concatenate([vector, to_pad], axis=0)

    vectors = np.concatenate([vectors, vector], axis=0)
  return vectors




def pad(xyz, dis_from_agent, upper_bound=51, for_granted=30, for_granted_close=40, step=3):


  sorted_idx = np.lexsort(dis_from_agent.reshape(1, -1))


  if len(xyz) < upper_bound:
    n_padding = upper_bound - len(xyz)
    pad_idx = np.concatenate([np.arange(len(xyz)), np.ones(n_padding)*len(xyz)-1])


  elif len(xyz) < for_granted + (upper_bound-for_granted) * step:
    extra_idx = np.random.randint(for_granted_close, len(xyz), upper_bound-for_granted_close)
    pad_idx = np.concatenate([np.arange(for_granted_close), extra_idx])


  else :
    pad_idx = np.concatenate([np.arange(for_granted), np.arange(for_granted, for_granted+(upper_bound-for_granted)*step, step)])
  # print(pad_idx)

  return sorted_idx[pad_idx.astype(int)].astype(int)



def vectorize_lanes(_lane_data):
  vectors = np.empty((0, 10))
  for i in range(len(_lane_data)):

    pad_idx = pad(_lane_data[i]['Lane_XYZ'], _lane_data[i]['dis_from_agent'])
    xyzs = _lane_data[i]['Lane_XYZ'][pad_idx][:-1]
    xyze = _lane_data[i]['Lane_XYZ'][pad_idx][1:]

    dir = np.expand_dims(_lane_data[i]['Lane_DIR'][pad_idx], -1)
    dir = np.concatenate([dir[:-1], dir[1:]], axis=-1).mean(axis=-1)

    typ = _lane_data[i]['Lane_TYPE'][pad_idx][:-1].T

    vector = np.vstack([xyzs[:, 0], xyzs[:, 1], xyzs[:, 2], xyze[:, 0], xyze[:, 1], xyze[:, 2], dir[:, 0], dir[:, 1], dir[:, 2], typ]).T

    vectors = np.vstack([vectors, vector])
  return vectors


def extract_features(i, XY, Velocity, Current_valid, Agents_val, Agent_type, YAWS, GT_XY, Future_valid, Tracks_to_predict,
                     lane_xyz, lane_valid, lane_dir, lane_id, lane_type,
                     sm, DISP, validate=True):

  xy = XY[i]
  velocity = Velocity[i]
  current_val = Current_valid[i]
  val = Agents_val[i]
  agent_type = Agent_type[i]
  yaw = YAWS[i]
  gt_xy = GT_XY[i]
  is_avail = Future_valid[i]
  predict = Tracks_to_predict[i]

  # Discard unwanted tracks
  if predict == 0 or current_val==0 or val.mean() < 1:
    return -1


  # Extract Agent Features:
  radius, agent_data = extract_agent_features(i, xy, velocity, yaw, gt_xy, is_avail, DISP)

  # Extract Object Features:
  obj_data = []
  for j in range(len(xy)):

    obj_xy = XY[j]
    obj_velocity = Velocity[j]
    obj_current_val = Current_valid[j]
    obj_val = Agents_val[j]
    obj_agent_type = Agent_type[j]
    obj_yaw = YAWS[j]
    obj_gt_xy = GT_XY[j]
    obj_is_avail = Future_valid[j]
    obj_predict = Tracks_to_predict[j]

    if i==j or obj_current_val==0 or DISP[i, j, -1] > radius:
      continue

    _obj_data = extract_object_features(xy, obj_xy, obj_velocity, obj_yaw, Agent_type[j])
    obj_data.append(_obj_data)

  lane_xy = lane_xyz[:, :2]
  dis_from_agent = (xy[-1]-lane_xy)**2
  dis_from_agent = np.sqrt(dis_from_agent.sum(axis=1))

  lane_data = []

  for id in range(len(sm)-1):
    start = sm[id]
    end = sm[id+1]
    min_dis = dis_from_agent[start : end].min()

    if min_dis*75 > radius or (end-start) < 5:
      continue

    _lane_xyz = lane_xyz[start:end]
    _lane_is_inter = np.zeros(end-start)
    _lane_dir = lane_dir[start:end]
    _lane_type = lane_type[start:end]

    _lane_data = {
          'Lane_XYZ' : _lane_xyz,
          'Lane_IS_INTER' : _lane_is_inter,
          'Lane_DIR' : _lane_dir,
          'Lane_TYPE' : _lane_type,
          'Lane_ID' : id,
          'dis_from_agent' : dis_from_agent[start:end],
      }

    lane_data.append(_lane_data)

  return agent_data, obj_data, lane_data



def load_data(scene_file, n_shards=1):
  dataset = tf.data.TFRecordDataset(
          [scene_file], num_parallel_reads=1
      )

  dataset = tf.data.TFRecordDataset(
        [scene_file], num_parallel_reads=1
    )
  if n_shards > 1:
      dataset = dataset.shard(n_shards, 0)

  # record = next(iter(dataset))
  data = []
  for record in dataset:
    example = tf.train.Example()
    example.ParseFromString(record.numpy())
    parsed = example.features.feature

    # Current Agent States
    Past_x = np.array(parsed["state/past/x"].float_list.value).reshape(128, 10)
    Past_y = np.array(parsed["state/past/y"].float_list.value).reshape(128, 10)
    Current_x = np.array(parsed["state/current/x"].float_list.value).reshape(128, 1)
    Current_y = np.array(parsed["state/current/y"].float_list.value).reshape(128, 1)
    XY = np.concatenate(
            (
                np.expand_dims(np.concatenate((Past_x, Current_x), axis=1), axis=-1),
                np.expand_dims(np.concatenate((Past_y, Current_y), axis=1), axis=-1),
            ),
            axis=-1,
        )


    Current_yaw = np.array(parsed["state/current/bbox_yaw"].float_list.value).reshape(128, 1)
    Past_yaw = np.array(parsed["state/past/bbox_yaw"].float_list.value).reshape(128, 10)
    YAWS = np.concatenate((Past_yaw, Current_yaw), axis=1)


    Past_velocity_x = np.array(parsed["state/past/velocity_x"].float_list.value).reshape(128, 10)
    Past_velocity_y = np.array(parsed["state/past/velocity_y"].float_list.value).reshape(128, 10)

    Current_velocity_x = np.array(parsed["state/current/velocity_x"].float_list.value).reshape(128, 1)
    Current_velocity_y = np.array(parsed["state/current/velocity_y"].float_list.value).reshape(128, 1)

    Velocity = np.concatenate(
            (
                np.expand_dims(np.concatenate((Past_velocity_x, Current_velocity_y), axis=1), axis=-1),
                np.expand_dims(np.concatenate((Past_velocity_x, Current_velocity_y), axis=1), axis=-1),
            ),
            axis=-1,
        )

    Past_valid = np.array(parsed["state/past/valid"].int64_list.value).reshape(128, 10)
    Current_valid = np.array(parsed["state/current/valid"].int64_list.value).reshape(128, 1)
    Agents_val = np.concatenate((Past_valid, Current_valid), axis=1)

    Agent_type = np.array(parsed["state/type"].float_list.value).reshape(128)


    # Future Agent states
    X_gt = np.array(parsed["state/future/x"].float_list.value).reshape(128, 80)
    Y_gt = np.array(parsed["state/future/y"].float_list.value).reshape(128, 80)
    Future_valid = np.array(parsed["state/future/valid"].int64_list.value).reshape(128, 80)
    Tracks_to_predict = np.array(parsed["state/tracks_to_predict"].int64_list.value).reshape(128)
    GT_XY = np.concatenate(
            (np.expand_dims(X_gt, axis=-1), np.expand_dims(Y_gt, axis=-1)), axis=-1
            )


    # Extract Lane info
    lane_dir = np.array(parsed["roadgraph_samples/dir"].float_list.value).reshape(-1, 3)
    lane_id = np.array(parsed["roadgraph_samples/id"].int64_list.value).reshape(-1, 1)
    lane_type = np.array(parsed["roadgraph_samples/type"].int64_list.value).reshape(-1, 1)
    lane_valid = np.array(parsed["roadgraph_samples/valid"].int64_list.value).reshape(-1, 1)
    lane_xyz = np.array(parsed["roadgraph_samples/xyz"].float_list.value).reshape(-1, 3)

    lane_valid = lane_valid.reshape(-1)
    lane_xyz = lane_xyz[lane_valid>0]
    lane_dir = lane_dir[lane_valid>0]
    lane_id = lane_id[lane_id>0]

    data.append([XY, Velocity, Current_valid, Agents_val, Agent_type, YAWS, \
            GT_XY, Future_valid, Tracks_to_predict, lane_xyz, lane_valid, lane_dir, lane_id, lane_type])
  return data