# python3 src/main.py --config=EMU_sc2 --env-config=sc2 with env_args.map_name=5m_vs_6m t_max=4050000
# python3 src/main.py --config=EMU_sc2 --env-config=sc2 with env_args.map_name=1c3s5z
import datetime
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6, 7' 
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.episode_buffer import Prioritized_ReplayBuffer
from components.transforms import OneHot
from utils.torch_utils import to_cuda
from modules.agents.LRN_KNN import LRU_KNN
from modules.agents.LRN_KNN_STATE import LRU_KNN_STATE
from components.episodic_memory_buffer import Episodic_memory_buffer

import numpy as np
import copy as cp
import random
import time
# nohup python3 src/main.py --config=EMC_sc2 --env-config=sc2 with env_args.map_name=5m_vs_6m > nohup_EMC_5m_vs_6m.out &
def run(_run, _config, _log):
    # os.environ['SC2PATH'] = '//EMU_pymarl/3rdparty/StarCraftII'
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"
    os.environ['SET_DEVICE'] = '5'
    set_device = os.getenv('SET_DEVICE')
    print('SUNMENGYAO____50____________________')
    print('cuda',th.version.cuda)
    # args.device = "cpu"
    if args.use_cuda and set_device != '-1':
        if set_device is None:
            args.device = "cuda"
        else:
            # args.device = f"cuda:{set_device}"
            args.device = th.device("cuda:{}".format(set_device))
            # args.device = "cuda:{2}"
    else:
        args.device = "cpu"
    # os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '2' 
    # args.device = th.device('cuda:2')

    print('SUNMENGYAO________________________args.device={}'.format(args.device))
    # setup loggers
    logger = Logger(_log)
    _log.info("device:{}".format(args.device))
    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs", args.env,
                                     args.env_args['map_name'])
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)
        tb_info_get = os.path.join("results", "tb_logs", args.env, args.env_args['map_name'], "{}").format(unique_token)
        _log.info("saving tb_logs to " + tb_info_get)

    # sacred is on by default
    logger.setup_sacred(_run) # 是一个用于设置 Sacred 日志记录框架的方法调用，将日志记录器与 Sacred 实验记录集成，以便将日志信息保存到 Sacred 实验记录中。

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def save_one_buffer(args, save_buffer, env_name, from_start=False):
    x_env_name = env_name
    if from_start:
        x_env_name += '_from_start/'
    path_name = '../../buffer/' + x_env_name + '/buffer_' + str(args.save_buffer_id) + '/'
    if os.path.exists(path_name):
        random_name = '../../buffer/' + x_env_name + '/buffer_' + str(random.randint(10, 1000)) + '/'
        os.rename(path_name, random_name)
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    save_buffer.save(path_name)


def run_sequential(args, logger):
    # setup loggers
    # pymarl自定义了一个utils.logging.Logger类的对象logger对ex的内置变量_run和_log进行封装，
    # 最终所有的实验结果通过 logger.log_stat(key, value, t, to_sacred=True) 记录在了./results/sacred/实验编号/info.json文件中。
    # 在整个实验中，logger主要对runner和learner两个对象所产生的实验数据进行了记录，包括训练数据和测试数据的如下内容：
    # 【runner对象】：
    # 一系列环境特定的实验数据，即env_info，在SC2中，包括"battle_won_mean"，"dead_allies_mean"， "dead_enemies_mean";
    # 训练相关的实验数据：包括"ep_length_mean"，"epsilon"，"return_mean"；
    # 【learner对象】：
    # 训练相关的实验数据："loss"，"grad_norm"，"td_error_abs" ，"q_taken_mean"，"target_mean"。

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger) # 创建了一个特定类型的运行器对象

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.episode_limit = env_info["episode_limit"]  # 最大回合数episode_limit
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.unit_dim = env_info["unit_dim"]
    args.obs_shape = env_info["obs_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "flag_win": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    env_name = args.env
    if env_name == 'sc2':
        env_name += '/' + args.env_args['map_name']

    if args.is_prioritized_buffer: # 优先级经验回放
        buffer = Prioritized_ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                                          args.prioritized_buffer_alpha,
                                          preprocess=preprocess,
                                          device="cpu" if args.buffer_cpu_only else args.device)
    else: # 经典的经验回放
        buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                              args.burn_in_period,
                              preprocess=preprocess,
                              device="cpu" if args.buffer_cpu_only else args.device)

    if args.is_save_buffer:
        save_buffer = ReplayBuffer(scheme, groups, args.save_buffer_size, env_info["episode_limit"] + 1,
                                   args.burn_in_period,
                                   preprocess=preprocess,
                                   device="cpu" if args.buffer_cpu_only else args.device)
    # 从预先保存的缓存中加载经验数据，以用于训练模型。（false）
    if args.is_batch_rl:
        assert (args.is_save_buffer == False)
        x_env_name = env_name
        if args.is_from_start:
            x_env_name += '_from_start/'
        path_name = '../../buffer/' + x_env_name + '/buffer_' + str(args.load_buffer_id) + '/'
        assert (os.path.exists(path_name) == True)
        buffer.load(path_name)
    # 创建episodic memory buffer
    if getattr(args, "use_emdqn", False):
        ec_buffer=Episodic_memory_buffer(args,scheme)
    else:
        ec_buffer=None 

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    mac.cuda()
    # Give runner the scheme
    if args.runner != 'offpolicy':
        runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    if args.learner=="fast_QLearner" or args.learner=="qplex_curiosity_vdn_learner" or args.learner=="max_q_learner":
        learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args, groups=groups)
    else:
        learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.runner == 'offpolicy':
        runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac, test_mac=learner.extrinsic_mac)

    if hasattr(args, "save_buffer") and args.save_buffer:
        learner.buffer = buffer

    if args.use_cuda:
        learner.cuda()
    # 在强化学习训练中加载之前保存的模型检查点（其实为空）
    if args.checkpoint_path != "":

        timesteps = [] # 存放检查点的时间步
        timestep_to_load = 0

        # 判断检查点路径是否为空（确实为空）
        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path（yaml中设置为空）
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():  # 假设这些数字代表保存模型时的时间步
                timesteps.append(int(name))  # 存放检查点的时间步？
        # 根据load_step的值来决定【加载】哪一个时间步的【模型】
        if args.load_step == 0:
            timestep_to_load = max(timesteps)   # 选择timesteps列表中的最大值（即最新的检查点）
        else:
            # 选择与load_step最接近的时间步
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)  # 加载模型
        runner.t_env = timestep_to_load  # 将runner.t_env设置为已加载模型的时间步，以便后续使用

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)  # 进行评估或保存回放
            return

    ''' ———————————————————————————————————————————————start training—————————————————————————————————————— '''
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0
    embedder_update_time = 0
    ec_buffer_stats_update_time=0   

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:  # runner.t_env为已加载模型的时间步
        # ① 更新replay buffer（is_batch_rl为false）
        if not args.is_batch_rl:
            # （1）交互获得一个完整的episode(不是测试模式)
            episode_batch = runner.run(test_mode=False)
            # （2）更新episodic memory buffer【基于algorithm2】
            if getattr(args, "use_emdqn", False):
                # 如果有state embedding structure
                if args.use_AEM == True:
                    # 更新经验缓冲区的统计信息 periodically update buffer statistics
                    if (runner.t_env - ec_buffer_stats_update_time >= args.ec_buffer_stats_update_interval) and (runner.t_env >= args.t_EC_update):
                        ec_buffer.update_ec_buffer_stats()
                        ec_buffer_stats_update_time = runner.t_env
                    # 更新episodic memory buffer，更新Ncall、Nxi
                    ec_buffer.update_ec_modified(episode_batch)
                # 没有启用AEM，则使用原始方法更新episodic memory buffer
                else:
                    ec_buffer.update_ec_original(episode_batch)
            #（3）更新普通buffer
            buffer.insert_episode_batch(episode_batch)
            # 如果设置了保存缓冲区的标志，则将episode_batch插入到另一个save_buffer缓冲区中
            if args.is_save_buffer:
                save_buffer.insert_episode_batch(episode_batch)
                if save_buffer.is_from_start and save_buffer.episodes_in_buffer == save_buffer.buffer_size:
                    save_buffer.is_from_start = False
                    save_one_buffer(args, save_buffer, env_name, from_start=True)
                    break
                if save_buffer.buffer_index % args.save_buffer_interval == 0 and os.name != 'nt':
                    print('current episodes_in_buffer: ', save_buffer.episodes_in_buffer)

        # ② 【进行模型训练】，并根据训练结果（如TD Error）动态调整样本的优先级
        for _ in range(args.num_circle): # args.num_circle 是预先设定的训练循环次数
            # 检查buffer是否有足够样本构成一个batch
            if buffer.can_sample(args.batch_size):
                if args.is_prioritized_buffer:  # 以优先级方式采样
                    sample_indices, episode_sample = buffer.sample(args.batch_size)
                else:  # 普通采样
                    episode_sample = buffer.sample(args.batch_size)
                # 如果是批量模式，根据采样的样本中的时间步来【更新环境时间（runner.t_env）】
                if args.is_batch_rl:
                    runner.t_env += int(th.sum(episode_sample['filled']).cpu().clone().detach().numpy()) // args.batch_size
                                    
                # 对不同长度的样本进行【填充】，以使它们具有相同的时间长度 Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args.device:
                    episode_sample.to(args.device)
                # 如果使用优先级buffer
                if args.is_prioritized_buffer:
                    if getattr(args, "use_emdqn", False):  # 如果使用EMDQN，则使用episodic memory buffer进行训练
                        # episode_sample是from普通buffer
                        # ec_buffer 是episodic control buffer
                        td_error = learner.train(episode_sample, runner.t_env, episode,ec_buffer=ec_buffer)
                    else:
                        td_error = learner.train(episode_sample, runner.t_env, episode)
                        buffer.update_priority(sample_indices, td_error)  # 基于TDerror更新样本优先级
                # 不使用优先级buffer
                else:
                    if getattr(args, "use_emdqn", False): # 如果使用EMDQN，则使用episodic memory buffer进行训练
                        td_error = learner.train(episode_sample, runner.t_env, episode, ec_buffer=ec_buffer)
                    else:
                        learner.train(episode_sample, runner.t_env, episode)

        # ③ 定期更新 state_embedding 和 prediction_network
        if (args.use_emdqn==True) and args.use_AEM and ( args.memory_emb_type == 2 or args.memory_emb_type == 3 ) and (runner.t_env - embedder_update_time >= args.ec_buffer_embedder_update_interval):
            embedder_update_time = runner.t_env # 更新嵌入器的最后更新时间为当前的环境时间
            emb_start_time = time.time()
            ec_buffer.train_embedder()   # 基于公式5训练state embedding structure
            ec_buffer.update_embedding() # 更新episodic memory buffer中的状态嵌入
            emb_end_time = time.time()
            total_time = emb_end_time - emb_start_time
            if os.name != 'nt': # 检查操作系统是否不是Windows（非NT内核）
                print("Processing time for memory embedding:", total_time )
            # 如果启用了额外的更新并且缓冲区已经构建了树结构
            if args.additional_update == True and ec_buffer.ec_buffer.build_tree == True:
                # re-update can fixate on current replay memory & can take long time
                if buffer.can_sample(args.buffer_size_update): # 检查缓冲区是否有足够的样本来进行额外的更新
                    add_train_start_time = time.time()
                    all_episode_sample = buffer.sample(args.buffer_size_update)  # 从缓冲区中采样指定数量的样本
                    ec_buffer.update_ec_modified(all_episode_sample)             # 使用采样的样本更新经验回放缓冲区
                    add_train_end_time = time.time()
                    total_time = add_train_end_time - add_train_start_time
                    if os.name != 'nt':
                        print("Processing time for additional memory update:", total_time )

        # ④ 测试
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)  # 计算需要执行的【测试运行次数】
        # 判断是否已经到了执行下一组测试运行的时间
        # runner.t_env 表示当前的环境时间（或步数），last_test_T 是上次测试运行的时间，args.test_interval 是两次测试之间的间隔时间
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0 :
            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()
            logger.log_stat("num_circle", args.num_circle, runner.t_env) # logger.log_stat(key, value, t, to_sacred=True) 记录在了./results/sacred/

            last_test_T = runner.t_env # 更新上次测试的时间为当前环境时间。
            for _ in range(n_test_runs):
                # 收集一批测试样本
                episode_sample = runner.run(test_mode=True)
                if args.mac == "offline_mac": # 【截断批次样本】，仅保留至最大填充时间步的数据
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]
                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)
                    # 使用收集的样本
                    learner.train(episode_sample, runner.t_env, episode, show_v=True)
        # ⑤ 保存模型
        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_folder = args.config_name + '_' + args.env_args['map_name']
            save_path = os.path.join(args.local_results_path, "models", save_folder, args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)

            #if args.double_q:
            #    os.makedirs(save_path + '_x', exist_ok=True)

            #if args.learner == 'curiosity_learner' or args.learner == 'curiosity_learner_new' or args.learner == 'qplex_curiosity_learner'\
            #        or args.learner == 'qplex_curiosity_rnd_learner' or args.learner =='qplex_rnd_history_curiosity_learner':
            #    os.makedirs(save_path + '/mac/', exist_ok=True)
            #    os.makedirs(save_path + '/extrinsic_mac/', exist_ok=True)
            #    os.makedirs(save_path + '/predict_mac/', exist_ok=True)
            #    if args.learner == 'curiosity_learner_new' or args.learner == 'qplex_curiosity_rnd_learner'or args.learner =='qplex_rnd_history_curiosity_learner':
            #        os.makedirs(save_path + '/rnd_predict_mac/', exist_ok=True)
            #        os.makedirs(save_path + '/rnd_target_mac/', exist_ok=True)

            #if args.learner == 'rnd_learner' or args.learner == 'rnd_learner2' or args.learner =='qplex_rnd_learner'\
            #        or args.learner =='qplex_rnd_history_learner' or args.learner =='qplex_rnd_emdqn_learner' :
            #    os.makedirs(save_path + '/mac/', exist_ok=True)
            #    os.makedirs(save_path + '/rnd_predict_mac/', exist_ok=True)
            #    os.makedirs(save_path + '/rnd_target_mac/', exist_ok=True)
            #if args.learner == 'qplex_curiosity_single_learner' or "qplex_curiosity_single_fast_learner":
            #    os.makedirs(save_path + '/mac/', exist_ok=True)
            #    os.makedirs(save_path + '/predict_mac/', exist_ok=True)
            #    os.makedirs(save_path + '/soft_update_target_mac/', exist_ok=True)

            # 保存在results/models/_3s_vs_5z/EMU_sc2__2024-03-11_02-24-55/34
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states            
            learner.save_models(save_path, ec_buffer)

        episode += args.batch_size_run * args.num_circle

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env
    # ————————————————————————————————————————————while止——————————————————————————————————————
    if args.is_save_buffer and save_buffer.is_from_start:
        save_buffer.is_from_start = False
        save_one_buffer(args, save_buffer, env_name, from_start=True)

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    # print('SUNMENGYAO________________________')
    # print(th.cuda.is_available())
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        print("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
