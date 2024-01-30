import os 
from colorama import Fore, Back, Style

# EXPERIMENT_NAME = "Argo-1"
SUPPORTED_EXPERIMENTS = [
    "Argo-1", 
    "Argo-Normalied", 
    "Argo-pad", 
    "Argo-avg", 
    "Argo-GNN-GNN"
]

# Data Paths
MAIN_DIR = "/main/Argoverse Dataset/"
STATS_DIR = "/content/stats"
TRAIN_DIR = os.path.join(MAIN_DIR, "train_interm")
VAL_DIR = os.path.join(MAIN_DIR, "val_interm")
TEST_DIR = os.path.join(MAIN_DIR, "test")

def OUT_DIR(EXPERIMENT_NAME): 
    if EXPERIMENT_NAME in SUPPORTED_EXPERIMENTS: 
        return os.path.join(MAIN_DIR, f"out/{EXPERIMENT_NAME}_out") 
    else : 
        raise Exception(f"Experiment {EXPERIMENT_NAME} not supported")
def TB_DIR(EXPERIMENT_NAME): 
    if EXPERIMENT_NAME in SUPPORTED_EXPERIMENTS: 
        return os.path.join(OUT_DIR(EXPERIMENT_NAME), "tb") 
    else : 
        raise Exception(f"Experiment {EXPERIMENT_NAME} not supported")
def CKPT_DIR(EXPERIMENT_NAME):
    if EXPERIMENT_NAME in SUPPORTED_EXPERIMENTS: 
        return os.path.join(OUT_DIR(EXPERIMENT_NAME), "ckpt") 
    else : 
        raise Exception(f"Experiment {EXPERIMENT_NAME} not supported")

# Stats Paths
LANE_MEANS = os.path.join(STATS_DIR, "lanes/lane_means.npy")
LANE_STDS = os.path.join(STATS_DIR, "lanes/lane_stds.npy")

AGENT_MEANS = os.path.join(STATS_DIR, "agents/agent_means.npy")
AGENT_STDS = os.path.join(STATS_DIR, "agents/agent_stds.npy")

OBJ_MEANS = os.path.join(STATS_DIR, "objects/obj_means.npy")
OBJ_STDS = os.path.join(STATS_DIR, "objects/obj_stds.npy")

GT_MEANS = os.path.join(STATS_DIR, "gt/gt_means.npy")
GT_STDS = os.path.join(STATS_DIR, "gt/gt_stds.npy")


# Configs
N_PAST = 60
N_FUTURE = 50
RADIUS_OFFSET = 1.5
VELOCITY_DISTANCE_RATIO = 10
TRAJ_DT = 0.1
LANE_DL = 1e13

ARGO_PAST_TIME = 5
ARGO_SAMPLE_RATE = 10


EPOCHS = 100
LOG_STEP = 10
STEPS_PER_EPOCH = 71

DEVICE = 'cpu'
CKPT_EPOCH = 10

TRAIN_BS = 64
VAL_BS = 128
LR = 1e-3

# Padding params
OBJ_PAD_LEN = 67
LANE_PAD_LEN = 145


OUT_ENC_DIM = 16
N_TRAJ = 6
OUT_DIM = 2 * N_TRAJ * N_FUTURE + N_TRAJ



# Data params
track_category_mapping = {
    0 : "TRACK_FRAGMENT",
    1 : "UNSCORED_TARCK",
    2 : "SCORED_TARCK",
    3 : "FOCAL_TARCK"
}

object_color_code = {
    'vehicle'           : "#ff1d00",
    'bus'               : "#e2e817",
    'pedestrian'        : "#40BF64",
    'motorcyclist'      : "#2dd294",
    'riderless_bicycle' : "#1549ea",
    'background'        : "#112222",
    'static'            : "#112222",
    'construction'      : "#112222",
    'unknown'           : "#112222",
}

map_object_type = {
    'vehicle'           : 0,
    'bus'               : 1,
    'bike'              : 2,
    'cyclist' 		 : 2,
    'pedestrian'        : 2,
    'motorcyclist'      : 3,
    'riderless_bicycle' : 4,
    'background'        : 5,
    'static'            : 6,
    'construction'      : 7,
    'unknown'           : 8,
}



# Architecture configs
AGENT_ENC = {
    'd_model' : 8,
    'n_heads' : 2,
    'hidden_dim' : 32,
    'hidden_nheads' : 4,
    'output_dim' : OUT_ENC_DIM
}

OBJ_ENC = {
    'd_model' : 11,
    'n_heads' : 1,
    'hidden_dim' : 32,
    'hidden_nheads' : 4,
    'output_dim' : OUT_ENC_DIM
}

LANE_ENC = {
    'd_model' : 9,
    'n_heads' : 3,
    'hidden_dim' : 32,
    'hidden_nheads' : 4,
    'output_dim' : OUT_ENC_DIM
}

GRAPH_AGENT_ENC = {
    'd_model' : 8,
    'n_heads' : 2,
    'd_hidden' : 32,
    'd_out' : OUT_ENC_DIM
}

GRAPH_OBJ_ENC = {
    'd_model' : 11,
    'n_heads' : 1,
    'd_hidden' : 32,
    'd_out' : OUT_ENC_DIM
}

GRAPH_LANE_ENC = {
    'd_model' : 9,
    'n_heads' : 3,
    'd_hidden' : 32,
    'd_out' : OUT_ENC_DIM
}


GLOBAL_ENC = {
    'in_dim' : 17,
    # 'n_heads' : 2,
    # 'hidden_dim' : 32,
    # 'hidden_nheads' : 2,
    'out_dim' : 64
}

GLOBAL_ENC_TRANS = {
    'd_model' : 17,
    'num_heads' : 1,
    'd_ff' : 32,
    'output_dim' : 64
}


DECODER = {
    'in_dim' : GLOBAL_ENC['out_dim'], # 64
    'hidden_dim' : 128, 
    'out_dim' : OUT_DIM
}


blk = Style.BRIGHT + Fore.BLACK
red = Style.BRIGHT + Fore.RED
blu = Style.BRIGHT + Fore.BLUE
grn_bck = Back.GREEN
res = Style.RESET_ALL