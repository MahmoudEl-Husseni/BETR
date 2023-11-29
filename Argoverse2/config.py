import os 
from colorama import Fore, Back, Style


MAIN_DIR = "/main/Argoverse Dataset/"
TRAIN_DIR = os.path.join(MAIN_DIR, "val__")
VAL_DIR = os.path.join(MAIN_DIR, "val__")
TEST_DIR = os.path.join(MAIN_DIR, "test")

OUT_DIR = os.path.join(MAIN_DIR, "out")
TB_DIR = os.path.join(OUT_DIR, "tb")
CKPT_DIR = os.path.join(OUT_DIR, "ckpt")


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
DEVICE = 'cuda'
CKPT_EPOCH = 10

TRAIN_BS = 64
VAL_BS = 128
LR = 1e-3



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

OUT_ENC_DIM = 16
N_TRAJ = 6
OUT_DIM = 2 * N_TRAJ * N_FUTURE + N_TRAJ


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

#  d_model, n_heads, hidden_dim, hidden_nheads, output_dim
GLOBAL_ENC = {
    'in_dim' : 17,
    # 'n_heads' : 2,
    # 'hidden_dim' : 32,
    # 'hidden_nheads' : 2,
    'out_dim' : 64
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