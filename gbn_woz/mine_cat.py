import time
import os
import json
import torch as th
import logging
import sys
sys.path.append("../")
from latent_dialog.utils import Pack, prepare_dirs_loggers, set_seed
import latent_dialog.corpora as corpora
from latent_dialog.data_loaders import BeliefDbDataLoaders
# from latent_dialog.evaluators import MultiWozEvaluator
from latent_dialog.evaluators_new import MultiWozEvaluator
from latent_dialog.models_task import GBNSCat
from latent_dialog.main import train, validate
import latent_dialog.domain as domain
from experiments_woz.dialog_utils import task_generate
import gbn_woz.config as cf

domain_name = 'object_division'
domain_info = domain.get_domain(domain_name)
# data_name = 'multiwoz_2.1'
data_name = 'norm-multi-woz'
# pre_dir = '../data/norm-multi-woz/'
pre_dir = '../data/' + data_name +'/'
config = Pack(
    seed = 10,
    train_path= pre_dir + 'train_dials.json',
    valid_path= pre_dir + 'val_dials.json',
    test_path= pre_dir + 'test_dials.json',
    max_vocab_size=1000,
    last_n_model=5,
    max_utt_len=50,
    max_dec_len=50,
    backward_size=2,
    batch_size=128,
    data_name = data_name,
    #gbn
    use_gbn=True, # use gbn or baseline
    # use_3gate=True, # use 3gate or 1gate. ps: must be under gbn
    gbn_rate=0.001,
    ignore_delx=True, #ignore delexed word like i.e.,[range]

    use_gpu=True,
    op='adam',
    init_lr=0.001,
    l2_norm=1e-05,
    momentum=0.0,
    grad_clip=5.0,
    dropout=0.5,
    max_epoch=100,
    embed_size=100,
    num_layers=1,
    utt_rnn_cell='gru',
    utt_cell_size=300,
    bi_utt_cell=True,
    enc_use_attn=True,
    dec_use_attn=True,
    dec_rnn_cell='lstm',
    dec_cell_size=300,
    dec_attn_mode='cat',
    y_size=10,
    k_size=20,
    beta = 0.001,
    simple_posterior=True,
    contextual_posterior=True,
    use_mi = False,
    use_pr = True,
    use_diversity = False,
    #
    beam_size=20,
    fix_batch = True,
    fix_train_batch=False,
    avg_type='word',
    print_step=300,
    # ckpt_step=354,
    # ckpt_step=1416,
    improve_threshold=0.996,
    patient_increase=2.0,
    save_model=True,
    early_stop=False,
    gen_type='greedy',
    preview_batch_num=None,
    k=domain_info.input_length(),
    init_range=0.1,
    pretrain_folder='gbn_cat128_w1000_True_T=[50, 30, 20]-2021-06-05-11-45-41',
    forward_only=False
)


set_seed(config.seed)
start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
stats_path = 'sys_config_log_model'
if data_name == 'multiwoz_2.1':
    stats_path = 'woz21_save_model'

# stats_path = '0506_saved_model'
if config.forward_only:
    saved_path = os.path.join(stats_path, config.pretrain_folder)
    config = Pack(json.load(open(os.path.join(saved_path, 'config.json'))))
    config['forward_only'] = True
else:
    if config.use_gbn == True:
        prefix = 'gbn_cat' + str(config.batch_size) + "_w" + str(config.max_vocab_size)
        prefix = prefix + '_' + str(config.ignore_delx) + "_T=" + str(cf.K)
    else:
        prefix = 'base_cat' + str(config.batch_size)
    saved_path = os.path.join(stats_path, prefix + '-' + start_time)
    # saved_path = os.path.join(stats_path, start_time+'-'+os.path.basename(__file__).split('.')[0])
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
config.saved_path = saved_path

prepare_dirs_loggers(config)
logger = logging.getLogger()
logger.info('[START]\n{}\n{}'.format(start_time, '=' * 30))
logger.info('vocab_size = {}'.format(config.max_vocab_size))
logger.info('ingore delexed word i.e., [range] = {}'.format(config.ignore_delx))
logger.info('topic size = {}'.format(cf.K))

config.saved_path = saved_path

# save configuration
with open(os.path.join(saved_path, 'config.json'), 'w') as f:
    json.dump(config, f, indent=4)  # sort_keys=True

corpus = corpora.NormMultiWozCorpus(config)
train_dial, val_dial, test_dial = corpus.get_corpus()
V = corpus.V_tm + len(corpus.tmvocab_ignore)
train_data = BeliefDbDataLoaders('Train', train_dial, V, corpus.tmvocab_ignore, config)
val_data = BeliefDbDataLoaders('Val', val_dial, V, corpus.tmvocab_ignore, config)
test_data = BeliefDbDataLoaders('Test', test_dial, V, corpus.tmvocab_ignore, config)
# old configs don't think about woz2.1

evaluator = MultiWozEvaluator(data_name)

model = GBNSCat(corpus, config)


if config.use_gpu:
    model.cuda()

best_epoch = None
if not config.forward_only:
    try:
        best_epoch = train(model, train_data, val_data, test_data, config, evaluator, gen=task_generate)
    except KeyboardInterrupt:
        print('Training stopped by keyboard.')
if best_epoch is None:
    model_ids = sorted([int(p.replace('-model', '')) for p in os.listdir(saved_path) if 'model' in p and 'rl' not in p])
    best_epoch = model_ids[-1]

# print("$$$ Load {}-model".format(best_epoch))
logger.info("$$$ Load {}-model".format(best_epoch))
# config.batch_size = 32
if not model.gbn:
    model.load_state_dict(th.load(os.path.join(saved_path, '{}-model'.format(best_epoch))))
else:
    state = th.load(os.path.join(saved_path, '{}-model'.format(best_epoch)))
    model.load_state_dict(state['model'])
    model.gbn_topic.Phi = state['phi']
    model.gbn_topic.Pi = state['pi']

logger.info("Forward Only Evaluation")

validate(model, val_data, config)
validate(model, test_data, config)

with open(os.path.join(saved_path, '{}_{}_valid_file.txt'.format(start_time, best_epoch)), 'w') as f:
    task_generate(model, val_data, config, evaluator, num_batch=None, dest_f=f)

with open(os.path.join(saved_path, '{}_{}_test_file.txt'.format(start_time, best_epoch)), 'w') as f:
    task_generate(model, test_data, config, evaluator, num_batch=None, dest_f=f)

end_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
print('[END]', end_time, '=' * 30)
