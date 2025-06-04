from functools import partial
# do not import SC2 in labtop
import socket
if 'MBP' not in socket.gethostname() and 'DESIGNARE' not in socket.gethostname():
    from .multiagentenv import MultiAgentEnv
    from .starcraft2.starcraft2 import StarCraft2Env
    from .stag_hunt.stag_hunt import StagHunt
    #, Matrix_game1Env, Matrix_game2Env, Matrix_game3Env, mmdp_game1Env, \
    #    spread_xEnv, spread_x2Env, TwoState
else:
    from .multiagentenv import MultiAgentEnv
    from .stag_hunt.stag_hunt import StagHunt
import sys
import os
# from .grf import  GoogleFootballEnv
from .grf import Academy_3_vs_1_with_Keeper, Academy_Counterattack_Easy, Academy_Counterattack_Hard
def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {
    "sc2": partial(env_fn, env=StarCraft2Env),
    "stag_hunt": partial(env_fn, env=StagHunt),
    # "academy_3_vs_1_with_keeper": partial(env_fn, env=GoogleFootballEnv),
    # "academy_counterattack_hard": partial(env_fn, env=GoogleFootballEnv),
} if 'MBP' not in socket.gethostname() and 'DESIGNARE' not in socket.gethostname() else {}

REGISTRY["academy_3_vs_1_with_keeper"]= partial(env_fn, env=Academy_3_vs_1_with_Keeper)
REGISTRY["academy_counterattack_easy"]= partial(env_fn, env=Academy_Counterattack_Easy)
REGISTRY["academy_counterattack_hard"]= partial(env_fn, env=Academy_Counterattack_Hard)
#if sys.platform == "linux":
#    os.environ.setdefault("SC2PATH",
#                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
