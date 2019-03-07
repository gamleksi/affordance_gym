# The root path to the trajectory vae models
TRAJ_MODELS_PATH = '/home/aleksi/mujoco_ws/src/affordance_gym/src/TrajectoryVAE/traj_models'

# The root path to the policy models
POLICY_MODELS_PATH = '/home/aleksi/catkin_ws/src/affordance_gym/policy_models'

# The root path to the VAED models
VAED_MODELS_PATH = '/home/aleksi/catkin_ws/src/affordance_gym/src/AffordanceVAED/vaed_models'

# The root path to the real camera results
KINECT_EXPERIMENTS_PATH = '/home/aleksi/catkin_ws/src/affordance_gym/kinect_experiments'


#  Camera Parameteters for training the policy in MuJoCo
#  These camera params are distributed within [-EPSILON, +EPSILON] when generating a perception training data

LOOK_AT = [0.70, 0.0, 0.0]
DISTANCE = 1.16
AZIMUTH = -90.
ELEVATION = -30

LOOK_AT_EPSILON = 0.05
DISTANCE_EPSILON = 0.05
ELEVATION_EPSILON = 2.
AZIMUTH_EPSILON = 2.


# The end effector position region when generating training data for the Trajectory VAE

LUMI_X_LIM = [0.4, 0.75]
LUMI_Y_LIM = [-0.20, 0.20]
LUMI_Z_LIM = [.3, .3]


# The cup position region

CUP_X_LIM = [0.4, 0.75]
CUP_Y_LIM = [-0.20, 0.20]

# The desired end effector position when there is not a cup on the table

NO_CUP_SHOWN_POSE = [0.42, -0.18]

# These cup names were used when doing real experiments

CUP_NAMES = ['rocket', 'karol', 'gray', 'can', 'blue', 'subway', 'yellow', 'mirror', 'red', 'other', 'stack2', 'stack3', 'bag']
