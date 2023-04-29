import time
import os
import sys
sys.path.append('../')
import json
import torch as th
from latent_dialog.utils import Pack, set_seed, prepare_dirs_loggers
from latent_dialog.corpora import NormMultiWozCorpus
from latent_dialog.agent_task import OfflineLatentRlAgent
from latent_dialog.main import OfflineTaskReinforce
from experiments_woz.dialog_utils import task_generate
from latent_dialog.models_task import GBNSCat
import logging


def main():
    start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    print('[START]', start_time, '='*30)
    # RL configuration
    env = 'gpu'

    choose="gbn"
    if choose=="gbn":
        # pretrained_folder = 'gbn_cat128_w1000_True_T=[50, 30, 20]-2021-05-09-22-48-59'
        # pretrained_model_id = 18
        pretrained_folder = 'gbn_cat128_w1000_True_T=[50, 30, 20]-2021-06-06-11-12-21'
        pretrained_model_id = 36
    else:
        pretrained_folder = 'base_cat128-2021-05-12-00-30-34'
        pretrained_model_id = 38

    # pre_dir = 'woz21_save_model'
    pre_dir = 'sys_config_log_model'
    exp_dir = os.path.join(pre_dir, pretrained_folder, 'rl-'+start_time)
    # exp_dir = os.path.join('sys_config_log_model', pretrained_folder, 'rl-'+start_time)
    # create exp folder
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    rl_config = Pack(
        train_path='../data/norm-multi-woz/train_dials.json',
        valid_path='../data/norm-multi-woz/val_dials.json',
        test_path='../data/norm-multi-woz/test_dials.json',
        # train_path='../data/multiwoz_2.1/train_dials.json',
        # valid_path='../data/multiwoz_2.1/val_dials.json',
        # test_path='../data/multiwoz_2.1/test_dials.json',
        sv_config_path = os.path.join(pre_dir, pretrained_folder, 'config.json'),
        sv_model_path = os.path.join(pre_dir, pretrained_folder, '{}-model'.format(pretrained_model_id)),

        rl_config_path = os.path.join(exp_dir, 'rl_config.json'),
        rl_model_path = os.path.join(exp_dir, 'rl_model'),

        ppl_best_model_path = os.path.join(exp_dir, 'ppl_best.model'),
        reward_best_model_path = os.path.join(exp_dir, 'reward_best.model'),
        record_path = exp_dir,
        record_freq = 200,
        sv_train_freq= 0,  # TODO pay attention to main.py, cuz it is also controlled there
        use_gpu = env == 'gpu',
        nepoch = 10,
        nepisode = 0,
        tune_pi_only=False,
        max_words = 100,
        temperature = 1.0,
        episode_repeat = 1.0,
        rl_lr = 0.01,
        momentum = 0.0,
        nesterov = False,
        gamma = 0.99,
        rl_clip = 5.0,
        random_seed = 100,
    )

    # save configuration
    with open(rl_config.rl_config_path, 'w') as f:
        json.dump(rl_config, f, indent=4)

    # set random seed
    set_seed(rl_config.random_seed)

    # load previous supervised learning configuration and corpus
    sv_config = Pack(json.load(open(rl_config.sv_config_path)))
    sv_config['dropout'] = 0.0
    sv_config['use_gpu'] = rl_config.use_gpu
    log_config = Pack(
        forward_only=False,
        saved_path = exp_dir
    )
    prepare_dirs_loggers(log_config)
    logger = logging.getLogger()
    logger.info('[START]\n{}\n{}'.format(start_time, '=' * 30))

    # V_size = sv_config["max_vocab_size"]
    # logger.info('vocab_size = {}'.format(V_size))
    # logger.info('ingore delexed word i.e., [range] = {}'.format(ignore_delx))
    corpus = NormMultiWozCorpus(sv_config)

    # TARGET AGENT
    sys_model = GBNSCat(corpus, sv_config)

    if sv_config.use_gpu:
        sys_model.cuda()

    if not sys_model.gbn:
        sys_model.load_state_dict(th.load(rl_config.sv_model_path, map_location=lambda storage, location: storage))

    else:
        state = th.load(rl_config.sv_model_path, map_location=lambda storage, location: storage)
        sys_model.load_state_dict(state['model'])
        sys_model.gbn_topic.Phi = state['phi']
        sys_model.gbn_topic.Pi = state['pi']
    # sys_model.load_state_dict(th.load(rl_config.sv_model_path, map_location=lambda storage, location: storage))
    sys_model.eval()
    sys = OfflineLatentRlAgent(sys_model, corpus, rl_config, name='System', tune_pi_only=rl_config.tune_pi_only)

    # start RL
    reinforce = OfflineTaskReinforce(sys, corpus, sv_config, sys_model, rl_config, task_generate)
    reinforce.run()

    end_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    print('[END]', end_time, '='*30)


if __name__ == '__main__':
    main()
