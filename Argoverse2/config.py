import os 
MAIN_DIR = "/main/argoverse2"
TRAIN_DIR = os.path.join(MAIN_DIR, "train")
VAL_DIR = os.path.join(MAIN_DIR, "val")
TEST_DIR = os.path.join(MAIN_DIR, "test")

# Configs

N_PAST = 60
N_FUTURE = 50
RADIUS_OFFSET = 1.5
TRAJ_DT = 0.1
LANE_DL = 1e13

ARGO_PAST_TIME = 5
ARGO_SAMPLE_RATE = 10


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