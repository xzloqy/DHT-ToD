import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from latent_dialog.base_models import BaseModel
from latent_dialog.corpora import SYS, EOS, PAD, BOS
from latent_dialog.utils import INT, FLOAT, LONG, Pack, cast_type
from latent_dialog.enc2dec.encoders import RnnUttEncoder
from latent_dialog.enc2dec.decoders import DecoderRNN, GEN, TEACH_FORCE
from latent_dialog.criterions import NLLEntropy, CatKLLoss, Entropy, NormKLLoss
from latent_dialog import nn_lib
import numpy as np
from gbn_woz.utils import initialize_Phi_Pi, update_Pi_Phi, Bow_sents
from latent_dialog.gbn import GBNModel
from gbn_woz_L1.gbn import GBNModel_L1
class SysPerfectBD2Word(BaseModel):
    def __init__(self, corpus, config):
        super(SysPerfectBD2Word, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size

        self.embedding = None
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)

        self.policy = nn.Sequential(nn.Linear(self.utt_encoder.output_size + self.db_size + self.bs_size,
                                              config.dec_cell_size), nn.Tanh(), nn.Dropout(config.dropout))

        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size,
                                  hidden_size=config.dec_cell_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=self.utt_encoder.output_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.bos_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.max_dec_len,
                                  embedding=self.embedding)

        self.nll = NLLEntropy(self.pad_id, config.avg_type)

    def forward(self, data_feed, mode, clf=False, gen_type='greedy', return_latent=False):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # pack attention context
        if self.config.dec_use_attn:
            attn_context = enc_outs
        else:
            attn_context = None

        # create decoder initial states
        dec_init_state = self.policy(th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)).unsqueeze(0)

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            # h_dec_init_state = utt_summary.squeeze(1).unsqueeze(0)
            dec_init_state = tuple([dec_init_state, dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               # (batch_size, response_size-1)
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               # (batch_size, max_ctx_len, ctx_cell_size)
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size)  # (batch_size, goal_nhid)
        if mode == GEN:
            return ret_dict, labels
        if return_latent:
            return Pack(nll=self.nll(dec_outputs, labels),
                        latent_action=dec_init_state)
        else:
            return Pack(nll=self.nll(dec_outputs, labels))

    def forward_rl(self, data_feed, max_words, temp=0.1):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # pack attention context
        if self.config.dec_use_attn:
            attn_context = enc_outs
        else:
            attn_context = None

        # create decoder initial states
        dec_init_state = self.policy(th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)).unsqueeze(0)

        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        # decode
        logprobs, outs = self.decoder.forward_rl(batch_size=batch_size,
                                                 dec_init_state=dec_init_state,
                                                 attn_context=attn_context,
                                                 vocab=self.vocab,
                                                 max_words=max_words,
                                                 temp=temp)
        return logprobs, outs


class SysPerfectBD2Cat(BaseModel):
    def __init__(self, corpus, config):
        super(SysPerfectBD2Cat, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        self.k_size = config.k_size
        self.y_size = config.y_size
        self.simple_posterior = config.simple_posterior
        self.contextual_posterior = config.contextual_posterior

        self.embedding = None
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)

        self.c2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size + self.db_size + self.bs_size,
                                          config.y_size, config.k_size, is_lstm=False)
        self.z_embedding = nn.Linear(self.y_size * self.k_size, config.dec_cell_size, bias=False)
        self.gumbel_connector = nn_lib.GumbelConnector(config.use_gpu)
        if not self.simple_posterior:
            if self.contextual_posterior:
                self.xc2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size * 2 + self.db_size + self.bs_size,
                                                   config.y_size, config.k_size, is_lstm=False)
            else:
                self.xc2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size, config.y_size, config.k_size, is_lstm=False)

        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size,
                                  hidden_size=config.dec_cell_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=config.dec_cell_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.bos_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.max_dec_len,
                                  embedding=self.embedding)

        self.nll = NLLEntropy(self.pad_id, config.avg_type)
        self.cat_kl_loss = CatKLLoss()
        self.entropy_loss = Entropy()
        self.log_uniform_y = Variable(th.log(th.ones(1) / config.k_size))
        self.eye = Variable(th.eye(self.config.y_size).unsqueeze(0))
        self.beta = self.config.beta if hasattr(self.config, 'beta') else 0.0
        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y.cuda()
            self.eye = self.eye.cuda()

    def valid_loss(self, loss, batch_cnt=None):
        if self.simple_posterior:
            total_loss = loss.nll
            if self.config.use_pr > 0.0:
                total_loss += self.beta * loss.pi_kl
        else:
            total_loss = loss.nll + loss.pi_kl

        if self.config.use_mi:
            total_loss += (loss.b_pr * self.beta)

        if self.config.use_diversity:
            total_loss += loss.diversity

        return total_loss

    def forward(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        # create decoder initial states
        if self.simple_posterior:
            logits_qy, log_qy = self.c2z(enc_last)
            sample_y = self.gumbel_connector(logits_qy, hard=mode==GEN)
            log_py = self.log_uniform_y
        else:
            logits_py, log_py = self.c2z(enc_last)
            # encode response and use posterior to find q(z|x, c)
            x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1))
            if self.contextual_posterior:
                logits_qy, log_qy = self.xc2z(th.cat([enc_last, x_h.squeeze(1)], dim=1))
            else:
                logits_qy, log_qy = self.xc2z(x_h.squeeze(1))

            # use prior at inference time, otherwise use posterior
            if mode == GEN or (use_py is not None and use_py is True):
                sample_y = self.gumbel_connector(logits_py, hard=False)
            else:
                sample_y = self.gumbel_connector(logits_qy, hard=True)

        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               # (batch_size, response_size-1)
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               # (batch_size, max_ctx_len, ctx_cell_size)
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size)  # (batch_size, goal_nhid)
        if mode == GEN:
            ret_dict['sample_z'] = sample_y
            ret_dict['log_qy'] = log_qy
            return ret_dict, labels

        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            # regularization qy to be uniform
            avg_log_qy = th.exp(log_qy.view(-1, self.config.y_size, self.config.k_size))
            avg_log_qy = th.log(th.mean(avg_log_qy, dim=0) + 1e-15)
            b_pr = self.cat_kl_loss(avg_log_qy, self.log_uniform_y, batch_size, unit_average=True)
            mi = self.entropy_loss(avg_log_qy, unit_average=True) - self.entropy_loss(log_qy, unit_average=True)
            pi_kl = self.cat_kl_loss(log_qy, log_py, batch_size, unit_average=True)
            q_y = th.exp(log_qy).view(-1, self.config.y_size, self.config.k_size)  # b
            p = th.pow(th.bmm(q_y, th.transpose(q_y, 1, 2)) - self.eye, 2)

            result['pi_kl'] = pi_kl

            result['diversity'] = th.mean(p)
            result['nll'] = self.nll(dec_outputs, labels)
            result['b_pr'] = b_pr
            result['mi'] = mi
            return result

    def forward_rl(self, data_feed, max_words, temp=0.1):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        # create decoder initial states
        if self.simple_posterior:
            logits_py, log_qy = self.c2z(enc_last)
        else:
            logits_py, log_qy = self.c2z(enc_last)

        qy = F.softmax(logits_py / temp, dim=1)  # (batch_size, vocab_size, )
        log_qy = F.log_softmax(logits_py, dim=1)  # (batch_size, vocab_size, )
        idx = th.multinomial(qy, 1).detach()
        logprob_sample_z = log_qy.gather(1, idx).view(-1, self.y_size)
        joint_logpz = th.sum(logprob_sample_z, dim=1)
        sample_y = cast_type(Variable(th.zeros(log_qy.size())), FLOAT, self.use_gpu)
        sample_y.scatter_(1, idx, 1.0)

        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        # decode
        logprobs, outs = self.decoder.forward_rl(batch_size=batch_size,
                                                 dec_init_state=dec_init_state,
                                                 attn_context=attn_context,
                                                 vocab=self.vocab,
                                                 max_words=max_words,
                                                 temp=0.1)
        return logprobs, outs, joint_logpz, sample_y


class SysPerfectBD2Gauss(BaseModel):
    def __init__(self, corpus, config):
        super(SysPerfectBD2Gauss, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        self.y_size = config.y_size
        self.simple_posterior = config.simple_posterior

        self.embedding = None
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)

        self.c2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size + self.db_size + self.bs_size,
                                          config.y_size, is_lstm=False)
        self.gauss_connector = nn_lib.GaussianConnector(self.use_gpu)
        self.z_embedding = nn.Linear(self.y_size, config.dec_cell_size)
        if not self.simple_posterior:
            self.xc2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size * 2 + self.db_size + self.bs_size,
                                               config.y_size, is_lstm=False)

        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size,
                                  hidden_size=config.dec_cell_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=config.dec_cell_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.bos_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.max_dec_len,
                                  embedding=self.embedding)

        self.nll = NLLEntropy(self.pad_id, config.avg_type)
        self.gauss_kl = NormKLLoss(unit_average=True)
        self.zero = cast_type(th.zeros(1), FLOAT, self.use_gpu)

    def valid_loss(self, loss, batch_cnt=None):
        if self.simple_posterior:
            total_loss = loss.nll
            if self.config.use_pr > 0.0:
                total_loss += self.config.beta * loss.pi_kl
        else:
            total_loss = loss.nll + loss.pi_kl

        return total_loss

    def forward(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)

        # create decoder initial states
        if self.simple_posterior:
            q_mu, q_logvar = self.c2z(enc_last)
            sample_z = self.gauss_connector(q_mu, q_logvar)
            p_mu, p_logvar = self.zero, self.zero
        else:
            p_mu, p_logvar = self.c2z(enc_last)
            # encode response and use posterior to find q(z|x, c)
            x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1))
            q_mu, q_logvar = self.xc2z(th.cat([enc_last, x_h.squeeze(1)], dim=1))

            # use prior at inference time, otherwise use posterior
            if mode == GEN or use_py:
                sample_z = self.gauss_connector(p_mu, p_logvar)
            else:
                sample_z = self.gauss_connector(q_mu, q_logvar)

        # pack attention context
        dec_init_state = self.z_embedding(sample_z.unsqueeze(0))
        attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size)  # (batch_size, goal_nhid)
        if mode == GEN:
            ret_dict['sample_z'] = sample_z
            return ret_dict, labels

        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            pi_kl = self.gauss_kl(q_mu, q_logvar, p_mu, p_logvar)
            result['pi_kl'] = pi_kl
            result['nll'] = self.nll(dec_outputs, labels)
            return result

    def gaussian_logprob(self, mu, logvar, sample_z):
        var = th.exp(logvar)
        constant = float(-0.5 * np.log(2*np.pi))
        logprob = constant - 0.5 * logvar - th.pow((mu-sample_z), 2) / (2.0*var)
        return logprob

    def forward_rl(self, data_feed, max_words, temp=0.1):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        # create decoder initial states
        p_mu, p_logvar = self.c2z(enc_last)

        sample_z = th.normal(p_mu, th.sqrt(th.exp(p_logvar))).detach()
        logprob_sample_z = self.gaussian_logprob(p_mu, self.zero, sample_z)
        joint_logpz = th.sum(logprob_sample_z, dim=1)

        # pack attention context
        dec_init_state = self.z_embedding(sample_z.unsqueeze(0))
        attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        # decode
        logprobs, outs = self.decoder.forward_rl(batch_size=batch_size,
                                                 dec_init_state=dec_init_state,
                                                 attn_context=attn_context,
                                                 vocab=self.vocab,
                                                 max_words=max_words,
                                                 temp=0.1)
        return logprobs, outs, joint_logpz, sample_z


class GBN2Gauss(BaseModel):
    def __init__(self, corpus, config):
        super(GBN2Gauss, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        self.y_size = config.y_size
        self.simple_posterior = config.simple_posterior

        self.embedding = None
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)
        self.c2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size + self.db_size + self.bs_size,
                                          config.y_size, is_lstm=False)


        self.gauss_connector = nn_lib.GaussianConnector(self.use_gpu)
        self.z_embedding = nn.Linear(self.y_size, config.dec_cell_size)
        if not self.simple_posterior:
            self.xc2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size * 2 + self.db_size + self.bs_size,
                                               config.y_size, is_lstm=False)

        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size,
                                  hidden_size=config.dec_cell_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=config.dec_cell_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.bos_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.max_dec_len,
                                  embedding=self.embedding)

        self.nll = NLLEntropy(self.pad_id, config.avg_type)
        self.gauss_kl = NormKLLoss(unit_average=True)
        self.zero = cast_type(th.zeros(1), FLOAT, self.use_gpu)

        # GBN
        self.gbn = config.use_gbn
        self.gbnrate = config.gbn_rate
        if self.gbn:

            h_size = self.utt_encoder.output_size + self.db_size + self.bs_size
            # h_size = self.utt_encoder.output_size
            self.h2zr = nn.Linear(100+h_size,h_size*2)
            self.t2c = nn.Linear(100,h_size,bias=False)
            self.h2c = nn.Linear(h_size,h_size)
            self.h2zr2 = nn.Linear(80+h_size,h_size*2)
            self.t2c2 = nn.Linear(80,h_size,bias=False)
            self.h2c2 = nn.Linear(h_size,h_size)
            self.h2zr3 = nn.Linear(50+h_size,h_size*2)
            self.t2c3 = nn.Linear(50,h_size,bias=False)
            self.h2c3 = nn.Linear(h_size,h_size)
            self.V = corpus.V_tm + len(corpus.tmvocab_ignore)
            self.tm_ignore = corpus.tmvocab_ignore
            self.Phi, self.Pi, self.NDot_Phi, self.NDot_Pi = initialize_Phi_Pi(corpus.V_tm)

            self.K_dim = (100,80,50)
            self.real_min = 2.2e-20
            self.state1 = nn.Sequential(nn.Linear(corpus.V_tm, 100, bias=False),
                                        nn.Sigmoid(),)
            self.state2 = nn.Sequential(nn.Linear(100, 80, bias=False),
                                        nn.Sigmoid(),)
            self.state3 = nn.Sequential(nn.Linear(80, 50, bias=False),
                                        nn.Sigmoid(),)

            self.Weilbullk = nn.ModuleList()
            self.Weilbullk.append(nn.Sequential(
                            nn.Linear(config.embed_size,1),
                            nn.Softplus()))
            self.Weilbullk.append(nn.Sequential(
                            nn.Linear(80,1),
                            nn.Softplus()))
            self.Weilbullk.append(nn.Sequential(
                            nn.Linear(50,1),
                            nn.Softplus()))
            self.Weilbulll = nn.ModuleList()
            self.Weilbulll.append(nn.Sequential(
                            nn.Linear(config.embed_size,100),
                            nn.Softplus()))
            self.Weilbulll.append(nn.Sequential(
                            nn.Linear(80,80),
                            nn.Softplus()))
            self.Weilbulll.append(nn.Sequential(
                            nn.Linear(50,50),
                            nn.Softplus()))

        self.HT = None

    def PhiPi(self):
        self.Phi_1, self.Phi_2, self.Phi_3 = self.Phi
        self.Pi_1, self.Pi_2, self.Pi_3 = self.Pi
        self.Phi_1 = th.from_numpy(self.Phi_1).to("cuda:0").float()
        self.Phi_2 = th.from_numpy(self.Phi_2).to("cuda:0").float()
        self.Phi_3 = th.from_numpy(self.Phi_3).to("cuda:0").float()
        self.Pi_1 = th.from_numpy(self.Pi_1).to("cuda:0").float()
        self.Pi_2 = th.from_numpy(self.Pi_2).to("cuda:0").float()
        self.Pi_3 = th.from_numpy(self.Pi_3).to("cuda:0").float()

    def valid_loss(self, loss, batch_cnt=None):
        if self.simple_posterior:
            if self.training and self.gbn:
                total_loss = loss.nll + loss.TM_loss
            else:
                total_loss = loss.nll
            if self.config.use_pr > 0.0:
                total_loss += self.config.beta * loss.pi_kl
        else:
            total_loss = loss.nll + loss.pi_kl

        return total_loss

    def Encoder_Weilbull(self,input_x, l):  # i = 0:T-1 , input_x N*V
        # feedforward
        k_tmp = self.Weilbullk[l](input_x)  # none * 1
        k_tmp = k_tmp.expand(-1, self.K_dim[l]) # reshpe   ????                                             # none * K_dim[i]
        k = th.clamp(k_tmp, min=self.real_min)
        lam = self.Weilbulll[l](input_x)  # none * K_dim[i]
        return k.T, lam.T

    def log_max_tf(self,input_x):
        return th.log(th.clamp(input_x, min=self.real_min))

    def reparameterization(self, Wei_shape, Wei_scale, l, Batch_Size):
        eps = th.Tensor(np.int32(self.K_dim[l]),Batch_Size).uniform_().to("cuda:0")  # K_dim[i] * none
        # eps = tf.ones(shape=[np.int32(self.K_dim[l]), Batch_Size], dtype=tf.float32) /2 # K_dim[i] * none
        theta = Wei_scale * th.pow(-self.log_max_tf(1 - eps), 1 / Wei_shape)
        theta_c = theta.T
        return theta, theta_c  # K*N    N*K

    def KL_GamWei(self, Gam_shape, Gam_scale, Wei_shape, Wei_scale):  # K_dim[i] * none
        eulergamma = 0.5772
        KL_Part1 = eulergamma * (1 - 1 / Wei_shape) + self.log_max_tf(Wei_scale / Wei_shape) + 1 + Gam_shape * self.log_max_tf(Gam_scale)
        KL_Part2 = -th.lgamma(Gam_shape) + (Gam_shape - 1) * (self.log_max_tf(Wei_scale) - eulergamma / Wei_shape)
        KL = KL_Part1 + KL_Part2 - Gam_scale * Wei_scale * th.exp(th.lgamma(1 + 1 / Wei_shape))
        return KL

    def topic_driven(self, topic_context):
        # context = th.tensor(topic_context).cuda()
        context = topic_context
        sent_J = context.size(1)
        self.LB = 0
        theta_1C_HT = []
        theta_2C_HT = []
        theta_3C_HT = []
        theta_1C_NORM = []
        theta_2C_NORM = []
        theta_3C_NORM = []
        # gbn_inputs = topic_context.reshape(-1,topic_context.size(-1))

        for j in range(sent_J):
            gbn_inputs = context[:,j,:]  ### N*V
            batch_size = gbn_inputs.size(0)

            state1 = self.state1(gbn_inputs.float())
            self.k_1, self.l_1 = self.Encoder_Weilbull(state1, 0)  # K*N,  K*N
            theta_1, theta_1c = self.reparameterization(self.k_1, self.l_1, 0, batch_size)  # K * N batch_size = 20
            
            state2 = self.state2(state1)
            self.k_2, self.l_2 = self.Encoder_Weilbull(state2, 1)  # K*N,  K*N
            theta_2, theta_2c = self.reparameterization(self.k_2, self.l_2, 1, batch_size)  # K * N batch_size = 20
        
            state3 = self.state3(state2)
            self.k_3, self.l_3 = self.Encoder_Weilbull(state3, 2)  # K*N,  K*N
            theta_3, theta_3c = self.reparameterization(self.k_3, self.l_3, 2, batch_size)  # K * N batch_size = 20
            if j==0:
                alpha_1_t = th.matmul(self.Phi_2, theta_2)
                alpha_2_t = th.matmul(self.Phi_3, theta_3)
                alpha_3_t = th.ones(self.K_dim[2], batch_size).to("cuda:0") # K * 1
            else:
                alpha_1_t = th.matmul(self.Phi_2, theta_2)+ th.matmul(self.Pi_1, theta_left_1)
                alpha_2_t = th.matmul(self.Phi_3, theta_3)+ th.matmul(self.Pi_2, theta_left_2)
                alpha_3_t = th.matmul(self.Pi_3, theta_left_3)
            L1_1_t = gbn_inputs.T * self.log_max_tf(th.matmul(self.Phi_1, theta_1)) - th.matmul(self.Phi_1, theta_1)  # - tf.lgamma( X_VN_t + 1)
            theta1_KL = self.KL_GamWei(alpha_1_t, th.tensor(1.0,device='cuda:0'), self.k_1, self.l_1).sum()
            # if theta1_KL > 0:
            #     theta1_KL = theta1_KL - theta1_KL
            theta2_KL = self.KL_GamWei(alpha_2_t, th.tensor(1.0,device='cuda:0'), self.k_2, self.l_2).sum()
            # if theta2_KL > 0:
            #     theta2_KL = theta2_KL - theta2_KL
            theta3_KL = self.KL_GamWei(alpha_3_t, th.tensor(1.0,device='cuda:0'), self.k_3, self.l_3).sum()
            # if theta3_KL > 0:
            #     theta3_KL = theta3_KL - theta3_KL
            self.LB = self.LB + (1 * L1_1_t.sum() + 0.1 * theta1_KL + 0.01 * theta2_KL + 0.001 * theta3_KL)/batch_size
            theta_left_1 = theta_1
            theta_left_2 = theta_2
            theta_left_3 = theta_3
            theta_1c_norm = theta_1c / th.clamp(theta_1c.max(1)[0].reshape(batch_size,1).repeat(1,100), min=self.real_min)
            theta_2c_norm = theta_2c / th.clamp(theta_2c.max(1)[0].reshape(batch_size,1).repeat(1,80), min=self.real_min)
            theta_3c_norm = theta_3c / th.clamp(theta_3c.max(1)[0].reshape(batch_size,1).repeat(1,50), min=self.real_min)
            theta_1C_HT.append(theta_1c)
            theta_2C_HT.append(theta_2c)
            theta_3C_HT.append(theta_3c)
            theta_1C_NORM.append(theta_1c_norm)
            theta_2C_NORM.append(theta_2c_norm)
            theta_3C_NORM.append(theta_3c_norm)
        self.theta_1C_HT = th.stack(theta_1C_HT, dim=0).transpose(0,2)
        self.theta_2C_HT = th.stack(theta_2C_HT, dim=0).transpose(0,2)
        self.theta_3C_HT = th.stack(theta_3C_HT, dim=0).transpose(0,2)
        self.theta_1C_NORM = th.stack(theta_1C_NORM, dim=0).transpose(0,2)
        self.theta_2C_NORM = th.stack(theta_2C_NORM, dim=0).transpose(0,2)
        self.theta_3C_NORM = th.stack(theta_3C_NORM, dim=0).transpose(0,2)
        # self.HT = th.cat([self.theta_1C_HT,self.theta_2C_HT,self.theta_3C_HT],0).reshape(230,-1).T
        # # DO
        # self.Phi, self.Pi, self.NDot_Phi, self.NDot_Pi = update_Pi_Phi(context, self.Phi, self.Pi, [self.theta_1C_HT,self.theta_2C_HT,self.theta_3C_HT], doc_num_batches, MBObserved, self.NDot_Phi, self.NDot_Pi)

    def GRU_theta_hidden(self,hidden,theta):
        h_size = hidden.size(-1)
        z, r = th.split(self.h2zr(th.cat([hidden,theta[0]],dim=-1)),h_size,dim=-1)
        z, r = th.sigmoid(z), th.sigmoid(r)
        # c = th.tanh(th.matmul(theta, self.gate_w) + tf.matmul((r * hidden), self.gate_u) +  self.gate_b)
        c = th.tanh(self.t2c(theta[0]) + self.h2c(r * hidden))
        hidden = (1-z)*hidden + z*c
        z, r = th.split(self.h2zr2(th.cat([hidden,theta[1]],dim=-1)),h_size,dim=-1)
        z, r = th.sigmoid(z), th.sigmoid(r)
        # c = th.tanh(th.matmul(theta, self.gate_w) + tf.matmul((r * hidden), self.gate_u) +  self.gate_b)
        c = th.tanh(self.t2c2(theta[1]) + self.h2c2(r * hidden))
        hidden = (1-z)*hidden + z*c
        z, r = th.split(self.h2zr3(th.cat([hidden,theta[2]],dim=-1)),h_size,dim=-1)
        z, r = th.sigmoid(z), th.sigmoid(r)
        # c = th.tanh(th.matmul(theta, self.gate_w) + tf.matmul((r * hidden), self.gate_u) +  self.gate_b)
        c = th.tanh(self.t2c3(theta[2]) + self.h2c3(r * hidden))
        hidden = (1-z)*hidden + z*c
        return hidden

    def GRU_theta_hidden_after(self,hidden,theta):
        h_size = hidden.size(-1)
        z, r = th.split(self.h2zr2(th.cat([hidden,theta],dim=-1)),h_size,dim=-1)
        z, r = th.sigmoid(z), th.sigmoid(r)
        # c = th.tanh(th.matmul(theta, self.gate_w) + tf.matmul((r * hidden), self.gate_u) +  self.gate_b)
        c = th.tanh(self.t2c2(theta) + self.h2c2(r * hidden))
        hidden = (1-z)*hidden + z*c
        return hidden

    def forward(self, data_feed, mode, doc_num_batches=0, MBObserved=0, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        if self.gbn:
            self.PhiPi()
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)
        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))
        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)

        # gbn
        # TODO 调整topic word context
        if self.gbn:
            topic_context = self.np2var(data_feed['sents'], LONG)
            if topic_context.size(0) < 32:
                topic_contextss = topic_context.reshape(topic_context.size(0),1,topic_context.size(-1))
            else:
                topic_contextss = topic_context.reshape(8,-1,topic_context.size(-1))
            self.topic_driven(topic_contextss)
            # self.HT = th.cat([self.theta_1C_HT,self.theta_2C_HT,self.theta_3C_HT],0).reshape(230,-1).T
            # self.HT = th.cat([self.theta_1C_NORM,self.theta_2C_NORM,self.theta_3C_NORM],0).reshape(230,-1).T
            self.HT = [self.theta_1C_NORM.T.reshape(batch_size,-1),self.theta_2C_NORM.T.reshape(batch_size,-1),self.theta_3C_NORM.T.reshape(batch_size,-1)]
            # DO
            enc_last = self.GRU_theta_hidden(enc_last,self.HT)

        if self.gbn and self.training:
            self.Phi, self.Pi, self.NDot_Phi, self.NDot_Pi = update_Pi_Phi(topic_contextss, self.Phi, self.Pi, [self.theta_1C_HT,self.theta_2C_HT,self.theta_3C_HT], doc_num_batches, MBObserved, self.NDot_Phi, self.NDot_Pi)
        else:
            self.LB = 0

        # create decoder initial states
        if self.simple_posterior:
            q_mu, q_logvar = self.c2z(enc_last)
            sample_z = self.gauss_connector(q_mu, q_logvar)
            p_mu, p_logvar = self.zero, self.zero
        else:
            p_mu, p_logvar = self.c2z(enc_last)
            # encode response and use posterior to find q(z|x, c)
            x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1))
            q_mu, q_logvar = self.xc2z(th.cat([enc_last, x_h.squeeze(1)], dim=1))

            # use prior at inference time, otherwise use posterior
            if mode == GEN or use_py:
                sample_z = self.gauss_connector(p_mu, p_logvar)
            else:
                sample_z = self.gauss_connector(q_mu, q_logvar)

        # pack attention context
        dec_init_state = self.z_embedding(sample_z.unsqueeze(0))
        # if self.gbn:
        #     dec_init_state = self.GRU_theta_hidden_after(dec_init_state.squeeze(),self.HT).unsqueeze(0)

        attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size
                                                               )  # (batch_size, goal_nhid)
        if mode == GEN:
            ret_dict['sample_z'] = sample_z
            return ret_dict, labels

        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            pi_kl = self.gauss_kl(q_mu, q_logvar, p_mu, p_logvar)
            result['pi_kl'] = pi_kl
            result['nll'] = self.nll(dec_outputs, labels)
            if self.training and self.gbn:
                result['TM_loss'] = -self.LB*self.gbnrate
            return result

    def gaussian_logprob(self, mu, logvar, sample_z):
        var = th.exp(logvar)
        constant = float(-0.5 * np.log(2*np.pi))
        logprob = constant - 0.5 * logvar - th.pow((mu-sample_z), 2) / (2.0*var)
        return logprob

    def forward_rl(self, data_feed, max_words, temp=0.1):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)

        if self.gbn:
            topic_context = self.np2var(data_feed['sents'], LONG)
            if topic_context.size(0) < 32:
                topic_contextss = topic_context.reshape(topic_context.size(0),1,topic_context.size(-1))
            else:
                topic_contextss = topic_context.reshape(8,-1,topic_context.size(-1))
            self.topic_driven(topic_contextss)
            # self.HT = th.cat([self.theta_1C_HT,self.theta_2C_HT,self.theta_3C_HT],0).reshape(230,-1).T
            # self.HT = th.cat([self.theta_1C_NORM,self.theta_2C_NORM,self.theta_3C_NORM],0).reshape(230,-1).T
            self.HT = [self.theta_1C_NORM.T.reshape(batch_size,-1),self.theta_2C_NORM.T.reshape(batch_size,-1),self.theta_3C_NORM.T.reshape(batch_size,-1)]
            # DO
            enc_last = self.GRU_theta_hidden(enc_last,self.HT)

        # create decoder initial states
        p_mu, p_logvar = self.c2z(enc_last)

        sample_z = th.normal(p_mu, th.sqrt(th.exp(p_logvar))).detach()
        logprob_sample_z = self.gaussian_logprob(p_mu, self.zero, sample_z)
        joint_logpz = th.sum(logprob_sample_z, dim=1)

        # pack attention context
        dec_init_state = self.z_embedding(sample_z.unsqueeze(0))
        attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        # decode
        logprobs, outs = self.decoder.forward_rl(batch_size=batch_size,
                                                 dec_init_state=dec_init_state,
                                                 attn_context=attn_context,
                                                 vocab=self.vocab,
                                                 max_words=max_words,
                                                 temp=0.1)
        return logprobs, outs, joint_logpz, sample_z


#gbn_cat
class GBN2Cat(BaseModel):

    def __init__(self, corpus, config):
        super(GBN2Cat, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        self.k_size = config.k_size
        self.y_size = config.y_size
        self.simple_posterior = config.simple_posterior
        self.contextual_posterior = config.contextual_posterior

        self.embedding = None
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)

        self.c2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size + self.db_size + self.bs_size,
                                          config.y_size, config.k_size, is_lstm=False)
        self.z_embedding = nn.Linear(self.y_size * self.k_size, config.dec_cell_size, bias=False)
        self.gumbel_connector = nn_lib.GumbelConnector(config.use_gpu)
        if not self.simple_posterior:
            if self.contextual_posterior:
                self.xc2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size * 2 + self.db_size + self.bs_size,
                                                   config.y_size, config.k_size, is_lstm=False)
            else:
                self.xc2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size, config.y_size, config.k_size, is_lstm=False)

        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size,
                                  hidden_size=config.dec_cell_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=config.dec_cell_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.bos_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.max_dec_len,
                                  embedding=self.embedding)

        self.nll = NLLEntropy(self.pad_id, config.avg_type)
        self.cat_kl_loss = CatKLLoss()
        self.entropy_loss = Entropy()
        self.log_uniform_y = Variable(th.log(th.ones(1) / config.k_size))
        self.eye = Variable(th.eye(self.config.y_size).unsqueeze(0))
        self.beta = self.config.beta if hasattr(self.config, 'beta') else 0.0
        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y.cuda()
            self.eye = self.eye.cuda()

        # GBN
        self.gbn = config.use_gbn
        self.gbnrate = config.gbn_rate
        if self.gbn:
            # 321=hidden_size
            h_size = self.utt_encoder.output_size + self.db_size + self.bs_size
            # h_size = self.utt_encoder.output_size
            self.h2zr = nn.Linear(100+h_size,h_size*2)
            self.t2c = nn.Linear(100,h_size,bias=False)
            self.h2c = nn.Linear(h_size,h_size)
            self.h2zr2 = nn.Linear(80+h_size,h_size*2)
            self.t2c2 = nn.Linear(80,h_size,bias=False)
            self.h2c2 = nn.Linear(h_size,h_size)
            self.h2zr3 = nn.Linear(50+h_size,h_size*2)
            self.t2c3 = nn.Linear(50,h_size,bias=False)
            self.h2c3 = nn.Linear(h_size,h_size)

            self.V = corpus.V_tm + len(corpus.tmvocab_ignore)
            self.tm_ignore = corpus.tmvocab_ignore
            self.Phi, self.Pi, self.NDot_Phi, self.NDot_Pi = initialize_Phi_Pi(corpus.V_tm)

            self.K_dim = (100,80,50)
            self.real_min = 2.2e-20
            self.state1 = nn.Sequential(nn.Linear(corpus.V_tm, 100, bias=False),
                                        nn.Sigmoid(),)
            self.state2 = nn.Sequential(nn.Linear(100, 80, bias=False),
                                        nn.Sigmoid(),)
            self.state3 = nn.Sequential(nn.Linear(80, 50, bias=False),
                                        nn.Sigmoid(),)

            self.Weilbullk = nn.ModuleList()
            self.Weilbullk.append(nn.Sequential(
                            nn.Linear(config.embed_size,1),
                            nn.Softplus()))
            self.Weilbullk.append(nn.Sequential(
                            nn.Linear(80,1),
                            nn.Softplus()))
            self.Weilbullk.append(nn.Sequential(
                            nn.Linear(50,1),
                            nn.Softplus()))
            self.Weilbulll = nn.ModuleList()
            self.Weilbulll.append(nn.Sequential(
                            nn.Linear(config.embed_size,100),
                            nn.Softplus()))
            self.Weilbulll.append(nn.Sequential(
                            nn.Linear(80,80),
                            nn.Softplus()))
            self.Weilbulll.append(nn.Sequential(
                            nn.Linear(50,50),
                            nn.Softplus()))
            # self.Weilbullk = nn.Sequential(
            #                  nn.Linear(config.embed_size,1),
            #                  nn.Softplus())
            # self.Weilbulll = nn.Sequential(
            #                  nn.Linear(config.embed_size,1),
            #                  nn.Softplus())
            # self.h1 = [];    self.h2 = [];    self.h3 = []
        self.HT = None

    def valid_loss(self, loss, batch_cnt=None):
        if self.simple_posterior:
            if self.training and self.gbn:
                total_loss = loss.nll + loss.TM_loss
            else:
                total_loss = loss.nll
            if self.config.use_pr > 0.0:
                total_loss += self.beta * loss.pi_kl
        else:
            total_loss = loss.nll + loss.pi_kl

        if self.config.use_mi:
            total_loss += (loss.b_pr * self.beta)

        if self.config.use_diversity:
            total_loss += loss.diversity

        return total_loss

    def PhiPi(self):
        self.Phi_1, self.Phi_2, self.Phi_3 = self.Phi
        self.Pi_1, self.Pi_2, self.Pi_3 = self.Pi
        self.Phi_1 = th.from_numpy(self.Phi_1).to("cuda:0").float()
        self.Phi_2 = th.from_numpy(self.Phi_2).to("cuda:0").float()
        self.Phi_3 = th.from_numpy(self.Phi_3).to("cuda:0").float()
        self.Pi_1 = th.from_numpy(self.Pi_1).to("cuda:0").float()
        self.Pi_2 = th.from_numpy(self.Pi_2).to("cuda:0").float()
        self.Pi_3 = th.from_numpy(self.Pi_3).to("cuda:0").float()

    def Encoder_Weilbull(self,input_x, l):  # i = 0:T-1 , input_x N*V
        # feedforward
        k_tmp = self.Weilbullk[l](input_x)  # none * 1
        k_tmp = k_tmp.expand(-1, self.K_dim[l]) # reshpe   ????                                             # none * K_dim[i]
        k = th.clamp(k_tmp, min=self.real_min)
        lam = self.Weilbulll[l](input_x)  # none * K_dim[i]
        return k.T, lam.T

    def log_max_tf(self,input_x):
        return th.log(th.clamp(input_x, min=self.real_min))

    def reparameterization(self, Wei_shape, Wei_scale, l, Batch_Size):
        eps = th.Tensor(np.int32(self.K_dim[l]),Batch_Size).uniform_().to("cuda:0")  # K_dim[i] * none
        # eps = tf.ones(shape=[np.int32(self.K_dim[l]), Batch_Size], dtype=tf.float32) /2 # K_dim[i] * none
        theta = Wei_scale * th.pow(-self.log_max_tf(1 - eps), 1 / Wei_shape)
        theta_c = theta.T
        return theta, theta_c  # K*N    N*K

    def KL_GamWei(self, Gam_shape, Gam_scale, Wei_shape, Wei_scale):  # K_dim[i] * none
        eulergamma = 0.5772
        KL_Part1 = eulergamma * (1 - 1 / Wei_shape) + self.log_max_tf(Wei_scale / Wei_shape) + 1 + Gam_shape * self.log_max_tf(Gam_scale)
        KL_Part2 = -th.lgamma(Gam_shape) + (Gam_shape - 1) * (self.log_max_tf(Wei_scale) - eulergamma / Wei_shape)
        KL = KL_Part1 + KL_Part2 - Gam_scale * Wei_scale * th.exp(th.lgamma(1 + 1 / Wei_shape))
        return KL

    def topic_driven(self, topic_context):
        # context = th.tensor(topic_context).cuda()
        context = topic_context
        sent_J = context.size(1)
        self.LB = 0
        theta_1C_HT = []
        theta_2C_HT = []
        theta_3C_HT = []
        theta_1C_NORM = []
        theta_2C_NORM = []
        theta_3C_NORM = []
        # gbn_inputs = topic_context.reshape(-1,topic_context.size(-1))

        for j in range(sent_J):
            gbn_inputs = context[:,j,:]  ### N*V
            batch_size = gbn_inputs.size(0)

            state1 = self.state1(gbn_inputs.float())
            self.k_1, self.l_1 = self.Encoder_Weilbull(state1, 0)  # K*N,  K*N
            theta_1, theta_1c = self.reparameterization(self.k_1, self.l_1, 0, batch_size)  # K * N batch_size = 20
            
            state2 = self.state2(state1)
            self.k_2, self.l_2 = self.Encoder_Weilbull(state2, 1)  # K*N,  K*N
            theta_2, theta_2c = self.reparameterization(self.k_2, self.l_2, 1, batch_size)  # K * N batch_size = 20
        
            state3 = self.state3(state2)
            self.k_3, self.l_3 = self.Encoder_Weilbull(state3, 2)  # K*N,  K*N
            theta_3, theta_3c = self.reparameterization(self.k_3, self.l_3, 2, batch_size)  # K * N batch_size = 20
            if j==0:
                alpha_1_t = th.matmul(self.Phi_2, theta_2)
                alpha_2_t = th.matmul(self.Phi_3, theta_3)
                alpha_3_t = th.ones(self.K_dim[2], batch_size).to("cuda:0") # K * 1
            else:
                alpha_1_t = th.matmul(self.Phi_2, theta_2)+ th.matmul(self.Pi_1, theta_left_1)
                alpha_2_t = th.matmul(self.Phi_3, theta_3)+ th.matmul(self.Pi_2, theta_left_2)
                alpha_3_t = th.matmul(self.Pi_3, theta_left_3)
            L1_1_t = gbn_inputs.T * self.log_max_tf(th.matmul(self.Phi_1, theta_1)) - th.matmul(self.Phi_1, theta_1)  # - tf.lgamma( X_VN_t + 1)
            theta1_KL = self.KL_GamWei(alpha_1_t, th.tensor(1.0,device='cuda:0'), self.k_1, self.l_1).sum()
            # if theta1_KL > 0:
            #     theta1_KL = theta1_KL - theta1_KL
            theta2_KL = self.KL_GamWei(alpha_2_t, th.tensor(1.0,device='cuda:0'), self.k_2, self.l_2).sum()
            # if theta2_KL > 0:
            #     theta2_KL = theta2_KL - theta2_KL
            theta3_KL = self.KL_GamWei(alpha_3_t, th.tensor(1.0,device='cuda:0'), self.k_3, self.l_3).sum()
            # if theta3_KL > 0:
            #     theta3_KL = theta3_KL - theta3_KL
            self.LB = self.LB + (1 * L1_1_t.sum() + 0.1 * theta1_KL + 0.01 * theta2_KL + 0.001 * theta3_KL)/batch_size
            theta_left_1 = theta_1
            theta_left_2 = theta_2
            theta_left_3 = theta_3
            theta_1c_norm = theta_1c / th.clamp(theta_1c.max(1)[0].reshape(batch_size,1).repeat(1,100), min=self.real_min)
            theta_2c_norm = theta_2c / th.clamp(theta_2c.max(1)[0].reshape(batch_size,1).repeat(1,80), min=self.real_min)
            theta_3c_norm = theta_3c / th.clamp(theta_3c.max(1)[0].reshape(batch_size,1).repeat(1,50), min=self.real_min)
            theta_1C_HT.append(theta_1c)
            theta_2C_HT.append(theta_2c)
            theta_3C_HT.append(theta_3c)
            theta_1C_NORM.append(theta_1c_norm)
            theta_2C_NORM.append(theta_2c_norm)
            theta_3C_NORM.append(theta_3c_norm)
        self.theta_1C_HT = th.stack(theta_1C_HT, dim=0).transpose(0,2)
        self.theta_2C_HT = th.stack(theta_2C_HT, dim=0).transpose(0,2)
        self.theta_3C_HT = th.stack(theta_3C_HT, dim=0).transpose(0,2)
        self.theta_1C_NORM = th.stack(theta_1C_NORM, dim=0).transpose(0,2)
        self.theta_2C_NORM = th.stack(theta_2C_NORM, dim=0).transpose(0,2)
        self.theta_3C_NORM = th.stack(theta_3C_NORM, dim=0).transpose(0,2)
        # self.HT = th.cat([self.theta_1C_HT,self.theta_2C_HT,self.theta_3C_HT],0).reshape(230,-1).T
        # # DO
        # self.Phi, self.Pi, self.NDot_Phi, self.NDot_Pi = update_Pi_Phi(context, self.Phi, self.Pi, [self.theta_1C_HT,self.theta_2C_HT,self.theta_3C_HT], doc_num_batches, MBObserved, self.NDot_Phi, self.NDot_Pi)

    def GRU_theta_hidden(self,hidden,theta):
        h_size = hidden.size(-1)
        z, r = th.split(self.h2zr(th.cat([hidden,theta[0]],dim=-1)),h_size,dim=-1)
        z, r = th.sigmoid(z), th.sigmoid(r)
        # c = th.tanh(th.matmul(theta, self.gate_w) + tf.matmul((r * hidden), self.gate_u) +  self.gate_b)
        c = th.tanh(self.t2c(theta[0]) + self.h2c(r * hidden))
        hidden = (1-z)*hidden + z*c
        z, r = th.split(self.h2zr2(th.cat([hidden,theta[1]],dim=-1)),h_size,dim=-1)
        z, r = th.sigmoid(z), th.sigmoid(r)
        # c = th.tanh(th.matmul(theta, self.gate_w) + tf.matmul((r * hidden), self.gate_u) +  self.gate_b)
        c = th.tanh(self.t2c2(theta[1]) + self.h2c2(r * hidden))
        hidden = (1-z)*hidden + z*c
        z, r = th.split(self.h2zr3(th.cat([hidden,theta[2]],dim=-1)),h_size,dim=-1)
        z, r = th.sigmoid(z), th.sigmoid(r)
        # c = th.tanh(th.matmul(theta, self.gate_w) + tf.matmul((r * hidden), self.gate_u) +  self.gate_b)
        c = th.tanh(self.t2c3(theta[2]) + self.h2c3(r * hidden))
        hidden = (1-z)*hidden + z*c
        return hidden

    def forward(self, data_feed, mode, doc_num_batches=0, MBObserved=0, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        if self.gbn:
            self.PhiPi()
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)

        # gbn
        # TODO 调整topic word context
        if self.gbn:
            topic_context = self.np2var(data_feed['sents'], LONG)
            if topic_context.size(0) < 32:
                topic_contextss = topic_context.reshape(topic_context.size(0),1,topic_context.size(-1))
            else:
                topic_contextss = topic_context.reshape(8,-1,topic_context.size(-1))
            self.topic_driven(topic_contextss)
            # self.HT = th.cat([self.theta_1C_HT,self.theta_2C_HT,self.theta_3C_HT],0).reshape(230,-1).T
            # self.HT = th.cat([self.theta_1C_NORM,self.theta_2C_NORM,self.theta_3C_NORM],0).reshape(230,-1).T
            self.HT = [self.theta_1C_NORM.T.reshape(batch_size,-1),self.theta_2C_NORM.T.reshape(batch_size,-1),self.theta_3C_NORM.T.reshape(batch_size,-1)]
            # DO
            enc_last = self.GRU_theta_hidden(enc_last,self.HT)

        if self.gbn and self.training:
            self.Phi, self.Pi, self.NDot_Phi, self.NDot_Pi = update_Pi_Phi(topic_contextss, self.Phi, self.Pi, [self.theta_1C_HT,self.theta_2C_HT,self.theta_3C_HT], doc_num_batches, MBObserved, self.NDot_Phi, self.NDot_Pi)
        else:
            self.LB = 0

        # create decoder initial states
        if self.simple_posterior:
            logits_qy, log_qy = self.c2z(enc_last)
            sample_y = self.gumbel_connector(logits_qy, hard=mode==GEN)
            log_py = self.log_uniform_y
        else:
            logits_py, log_py = self.c2z(enc_last)
            # encode response and use posterior to find q(z|x, c)
            x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1))
            if self.contextual_posterior:
                logits_qy, log_qy = self.xc2z(th.cat([enc_last, x_h.squeeze(1)], dim=1))
            else:
                logits_qy, log_qy = self.xc2z(x_h.squeeze(1))

            # use prior at inference time, otherwise use posterior
            if mode == GEN or (use_py is not None and use_py is True):
                sample_y = self.gumbel_connector(logits_py, hard=False)
            else:
                sample_y = self.gumbel_connector(logits_qy, hard=True)

        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               # (batch_size, response_size-1)
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               # (batch_size, max_ctx_len, ctx_cell_size)
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size)  # (batch_size, goal_nhid)
        if mode == GEN:
            ret_dict['sample_z'] = sample_y
            ret_dict['log_qy'] = log_qy
            return ret_dict, labels

        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            # regularization qy to be uniform
            avg_log_qy = th.exp(log_qy.view(-1, self.config.y_size, self.config.k_size))
            avg_log_qy = th.log(th.mean(avg_log_qy, dim=0) + 1e-15)
            b_pr = self.cat_kl_loss(avg_log_qy, self.log_uniform_y, batch_size, unit_average=True)
            mi = self.entropy_loss(avg_log_qy, unit_average=True) - self.entropy_loss(log_qy, unit_average=True)
            pi_kl = self.cat_kl_loss(log_qy, log_py, batch_size, unit_average=True)
            q_y = th.exp(log_qy).view(-1, self.config.y_size, self.config.k_size)  # b
            p = th.pow(th.bmm(q_y, th.transpose(q_y, 1, 2)) - self.eye, 2)

            result['pi_kl'] = pi_kl

            result['diversity'] = th.mean(p)
            result['nll'] = self.nll(dec_outputs, labels)
            result['b_pr'] = b_pr
            result['mi'] = mi
            if self.training and self.gbn:
                result['TM_loss'] = -self.LB*self.gbnrate
            return result

    def forward_rl(self, data_feed, max_words, temp=0.1):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        if self.gbn:
            topic_context = self.np2var(data_feed['sents'], LONG)
            if topic_context.size(0) < 32:
                topic_contextss = topic_context.reshape(topic_context.size(0),1,topic_context.size(-1))
            else:
                topic_contextss = topic_context.reshape(8,-1,topic_context.size(-1))
            self.topic_driven(topic_contextss)
            # self.HT = th.cat([self.theta_1C_HT,self.theta_2C_HT,self.theta_3C_HT],0).reshape(230,-1).T
            # self.HT = th.cat([self.theta_1C_NORM,self.theta_2C_NORM,self.theta_3C_NORM],0).reshape(230,-1).T
            self.HT = [self.theta_1C_NORM.T.reshape(batch_size,-1),self.theta_2C_NORM.T.reshape(batch_size,-1),self.theta_3C_NORM.T.reshape(batch_size,-1)]
            # DO
            enc_last = self.GRU_theta_hidden(enc_last,self.HT)

        # create decoder initial states
        if self.simple_posterior:
            logits_py, log_qy = self.c2z(enc_last)
        else:
            logits_py, log_qy = self.c2z(enc_last)

        qy = F.softmax(logits_py / temp, dim=1)  # (batch_size, vocab_size, )
        log_qy = F.log_softmax(logits_py, dim=1)  # (batch_size, vocab_size, )
        idx = th.multinomial(qy, 1).detach()
        logprob_sample_z = log_qy.gather(1, idx).view(-1, self.y_size)
        joint_logpz = th.sum(logprob_sample_z, dim=1)
        sample_y = cast_type(Variable(th.zeros(log_qy.size())), FLOAT, self.use_gpu)
        sample_y.scatter_(1, idx, 1.0)

        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        # decode
        logprobs, outs = self.decoder.forward_rl(batch_size=batch_size,
                                                 dec_init_state=dec_init_state,
                                                 attn_context=attn_context,
                                                 vocab=self.vocab,
                                                 max_words=max_words,
                                                 temp=0.1)
        return logprobs, outs, joint_logpz, sample_y







#simple sole gbn module
class GBNSCat(BaseModel):
    def __init__(self, corpus, config):
        super(GBNSCat, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        self.k_size = config.k_size
        self.y_size = config.y_size
        self.simple_posterior = config.simple_posterior
        self.contextual_posterior = config.contextual_posterior

        self.embedding = None
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)

        self.c2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size + self.db_size + self.bs_size,
                                          config.y_size, config.k_size, is_lstm=False)
        self.z_embedding = nn.Linear(self.y_size * self.k_size, config.dec_cell_size, bias=False)
        self.gumbel_connector = nn_lib.GumbelConnector(config.use_gpu)
        if not self.simple_posterior:
            if self.contextual_posterior:
                self.xc2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size * 2 + self.db_size + self.bs_size,
                                                   config.y_size, config.k_size, is_lstm=False)
            else:
                self.xc2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size, config.y_size, config.k_size, is_lstm=False)

        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size,
                                  hidden_size=config.dec_cell_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=config.dec_cell_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.bos_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.max_dec_len,
                                  embedding=self.embedding)

        self.nll = NLLEntropy(self.pad_id, config.avg_type)
        self.cat_kl_loss = CatKLLoss()
        self.entropy_loss = Entropy()
        self.log_uniform_y = Variable(th.log(th.ones(1) / config.k_size))
        self.eye = Variable(th.eye(self.config.y_size).unsqueeze(0))
        self.beta = self.config.beta if hasattr(self.config, 'beta') else 0.0
        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y.cuda()
            self.eye = self.eye.cuda()

        # GBN
        self.gbn = config.use_gbn
        self.gbnrate = config.gbn_rate
        if self.gbn:
            h_size = self.utt_encoder.output_size + self.db_size + self.bs_size
            self.gbn_topic = GBNModel(config, h_size, corpus.V_tm)
        self.HT = None

    def valid_loss(self, loss, batch_cnt=None):
        if self.simple_posterior:
            if self.training and self.gbn:
                total_loss = loss.nll + loss.TM_loss
            else:
                total_loss = loss.nll
            if self.config.use_pr > 0.0:
                total_loss += self.beta * loss.pi_kl
        else:
            total_loss = loss.nll + loss.pi_kl

        if self.config.use_mi:
            total_loss += (loss.b_pr * self.beta)

        if self.config.use_diversity:
            total_loss += loss.diversity

        return total_loss

    def Encoder_Weilbull(self,input_x, l):  # i = 0:T-1 , input_x N*V
        # feedforward
        k_tmp = self.Weilbullk[l](input_x)  # none * 1
        k_tmp = k_tmp.expand(-1, self.K_dim[l]) # reshpe   ????                                             # none * K_dim[i]
        k = th.clamp(k_tmp, min=self.real_min)
        lam = self.Weilbulll[l](input_x)  # none * K_dim[i]
        return k.T, lam.T

    def log_max_tf(self,input_x):
        return th.log(th.clamp(input_x, min=self.real_min))

    def reparameterization(self, Wei_shape, Wei_scale, l, Batch_Size):
        eps = th.Tensor(np.int32(self.K_dim[l]),Batch_Size).uniform_().to("cuda:0")  # K_dim[i] * none
        # eps = tf.ones(shape=[np.int32(self.K_dim[l]), Batch_Size], dtype=tf.float32) /2 # K_dim[i] * none
        theta = Wei_scale * th.pow(-self.log_max_tf(1 - eps), 1 / Wei_shape)
        theta_c = theta.T
        return theta, theta_c  # K*N    N*K

    def KL_GamWei(self, Gam_shape, Gam_scale, Wei_shape, Wei_scale):  # K_dim[i] * none
        eulergamma = 0.5772
        KL_Part1 = eulergamma * (1 - 1 / Wei_shape) + self.log_max_tf(Wei_scale / Wei_shape) + 1 + Gam_shape * self.log_max_tf(Gam_scale)
        KL_Part2 = -th.lgamma(Gam_shape) + (Gam_shape - 1) * (self.log_max_tf(Wei_scale) - eulergamma / Wei_shape)
        KL = KL_Part1 + KL_Part2 - Gam_scale * Wei_scale * th.exp(th.lgamma(1 + 1 / Wei_shape))
        return KL

    def topic_driven(self, topic_context):
        # context = th.tensor(topic_context).cuda()
        context = topic_context
        sent_J = context.size(1)
        self.LB = 0
        theta_1C_HT = []
        theta_2C_HT = []
        theta_3C_HT = []
        theta_1C_NORM = []
        theta_2C_NORM = []
        theta_3C_NORM = []
        # gbn_inputs = topic_context.reshape(-1,topic_context.size(-1))

        for j in range(sent_J):
            gbn_inputs = context[:,j,:]  ### N*V
            batch_size = gbn_inputs.size(0)

            state1 = self.state1(gbn_inputs.float())
            self.k_1, self.l_1 = self.Encoder_Weilbull(state1, 0)  # K*N,  K*N
            theta_1, theta_1c = self.reparameterization(self.k_1, self.l_1, 0, batch_size)  # K * N batch_size = 20
            
            state2 = self.state2(state1)
            self.k_2, self.l_2 = self.Encoder_Weilbull(state2, 1)  # K*N,  K*N
            theta_2, theta_2c = self.reparameterization(self.k_2, self.l_2, 1, batch_size)  # K * N batch_size = 20
        
            state3 = self.state3(state2)
            self.k_3, self.l_3 = self.Encoder_Weilbull(state3, 2)  # K*N,  K*N
            theta_3, theta_3c = self.reparameterization(self.k_3, self.l_3, 2, batch_size)  # K * N batch_size = 20
            if j==0:
                alpha_1_t = th.matmul(self.Phi_2, theta_2)
                alpha_2_t = th.matmul(self.Phi_3, theta_3)
                alpha_3_t = th.ones(self.K_dim[2], batch_size).to("cuda:0") # K * 1
            else:
                alpha_1_t = th.matmul(self.Phi_2, theta_2)+ th.matmul(self.Pi_1, theta_left_1)
                alpha_2_t = th.matmul(self.Phi_3, theta_3)+ th.matmul(self.Pi_2, theta_left_2)
                alpha_3_t = th.matmul(self.Pi_3, theta_left_3)
            L1_1_t = gbn_inputs.T * self.log_max_tf(th.matmul(self.Phi_1, theta_1)) - th.matmul(self.Phi_1, theta_1)  # - tf.lgamma( X_VN_t + 1)
            theta1_KL = self.KL_GamWei(alpha_1_t, th.tensor(1.0,device='cuda:0'), self.k_1, self.l_1).sum()
            # if theta1_KL > 0:
            #     theta1_KL = theta1_KL - theta1_KL
            theta2_KL = self.KL_GamWei(alpha_2_t, th.tensor(1.0,device='cuda:0'), self.k_2, self.l_2).sum()
            # if theta2_KL > 0:
            #     theta2_KL = theta2_KL - theta2_KL
            theta3_KL = self.KL_GamWei(alpha_3_t, th.tensor(1.0,device='cuda:0'), self.k_3, self.l_3).sum()
            # if theta3_KL > 0:
            #     theta3_KL = theta3_KL - theta3_KL
            self.LB = self.LB + (1 * L1_1_t.sum() + 0.1 * theta1_KL + 0.01 * theta2_KL + 0.001 * theta3_KL)/batch_size
            theta_left_1 = theta_1
            theta_left_2 = theta_2
            theta_left_3 = theta_3
            theta_1c_norm = theta_1c / th.clamp(theta_1c.max(1)[0].reshape(batch_size,1).repeat(1,100), min=self.real_min)
            theta_2c_norm = theta_2c / th.clamp(theta_2c.max(1)[0].reshape(batch_size,1).repeat(1,80), min=self.real_min)
            theta_3c_norm = theta_3c / th.clamp(theta_3c.max(1)[0].reshape(batch_size,1).repeat(1,50), min=self.real_min)
            theta_1C_HT.append(theta_1c)
            theta_2C_HT.append(theta_2c)
            theta_3C_HT.append(theta_3c)
            theta_1C_NORM.append(theta_1c_norm)
            theta_2C_NORM.append(theta_2c_norm)
            theta_3C_NORM.append(theta_3c_norm)
        self.theta_1C_HT = th.stack(theta_1C_HT, dim=0).transpose(0,2)
        self.theta_2C_HT = th.stack(theta_2C_HT, dim=0).transpose(0,2)
        self.theta_3C_HT = th.stack(theta_3C_HT, dim=0).transpose(0,2)
        self.theta_1C_NORM = th.stack(theta_1C_NORM, dim=0).transpose(0,2)
        self.theta_2C_NORM = th.stack(theta_2C_NORM, dim=0).transpose(0,2)
        self.theta_3C_NORM = th.stack(theta_3C_NORM, dim=0).transpose(0,2)
        # self.HT = th.cat([self.theta_1C_HT,self.theta_2C_HT,self.theta_3C_HT],0).reshape(230,-1).T
        # # DO
        # self.Phi, self.Pi, self.NDot_Phi, self.NDot_Pi = update_Pi_Phi(context, self.Phi, self.Pi, [self.theta_1C_HT,self.theta_2C_HT,self.theta_3C_HT], doc_num_batches, MBObserved, self.NDot_Phi, self.NDot_Pi)

    def GRU_theta_hidden(self,hidden,theta):
        h_size = hidden.size(-1)
        z, r = th.split(self.h2zr(th.cat([hidden,theta[0]],dim=-1)),h_size,dim=-1)
        z, r = th.sigmoid(z), th.sigmoid(r)
        # c = th.tanh(th.matmul(theta, self.gate_w) + tf.matmul((r * hidden), self.gate_u) +  self.gate_b)
        c = th.tanh(self.t2c(theta[0]) + self.h2c(r * hidden))
        hidden = (1-z)*hidden + z*c
        z, r = th.split(self.h2zr2(th.cat([hidden,theta[1]],dim=-1)),h_size,dim=-1)
        z, r = th.sigmoid(z), th.sigmoid(r)
        # c = th.tanh(th.matmul(theta, self.gate_w) + tf.matmul((r * hidden), self.gate_u) +  self.gate_b)
        c = th.tanh(self.t2c2(theta[1]) + self.h2c2(r * hidden))
        hidden = (1-z)*hidden + z*c
        z, r = th.split(self.h2zr3(th.cat([hidden,theta[2]],dim=-1)),h_size,dim=-1)
        z, r = th.sigmoid(z), th.sigmoid(r)
        # c = th.tanh(th.matmul(theta, self.gate_w) + tf.matmul((r * hidden), self.gate_u) +  self.gate_b)
        c = th.tanh(self.t2c3(theta[2]) + self.h2c3(r * hidden))
        hidden = (1-z)*hidden + z*c
        return hidden

    def forward(self, data_feed, mode, doc_num_batches=0, MBObserved=0, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        if self.gbn:
            self.gbn_topic.PhiPi()
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)

        # gbn
        # TODO 调整topic word context
        if self.gbn:
            topic_context = self.np2var(data_feed['sents'], LONG)
            if topic_context.size(0) < self.config.batch_size:
                topic_contextss = topic_context.reshape(topic_context.size(0),1,topic_context.size(-1))
            else:
                topic_contextss = topic_context.reshape(int(self.config.batch_size/4),-1,topic_context.size(-1))
            # self.topic_driven(topic_contextss)
            self.LB = self.gbn_topic.topic_driven(topic_contextss, doc_num_batches, MBObserved)
            # self.HT = th.cat([self.theta_1C_HT,self.theta_2C_HT,self.theta_3C_HT],0).reshape(230,-1).T
            # self.HT = th.cat([self.theta_1C_NORM,self.theta_2C_NORM,self.theta_3C_NORM],0).reshape(230,-1).T
            # self.HT = [self.theta_1C_NORM.T.reshape(batch_size,-1),self.theta_2C_NORM.T.reshape(batch_size,-1),self.theta_3C_NORM.T.reshape(batch_size,-1)]
            # DO
            enc_last = self.gbn_topic.GRU_theta_hidden(enc_last)

        if self.gbn and self.training:
            self.gbn_topic.update()
        else:
            self.LB = 0

        # create decoder initial states
        if self.simple_posterior:
            logits_qy, log_qy = self.c2z(enc_last)
            sample_y = self.gumbel_connector(logits_qy, hard=mode==GEN)
            log_py = self.log_uniform_y
        else:
            logits_py, log_py = self.c2z(enc_last)
            # encode response and use posterior to find q(z|x, c)
            x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1))
            if self.contextual_posterior:
                logits_qy, log_qy = self.xc2z(th.cat([enc_last, x_h.squeeze(1)], dim=1))
            else:
                logits_qy, log_qy = self.xc2z(x_h.squeeze(1))

            # use prior at inference time, otherwise use posterior
            if mode == GEN or (use_py is not None and use_py is True):
                sample_y = self.gumbel_connector(logits_py, hard=False)
            else:
                sample_y = self.gumbel_connector(logits_qy, hard=True)

        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               # (batch_size, response_size-1)
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               # (batch_size, max_ctx_len, ctx_cell_size)
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size)  # (batch_size, goal_nhid)
        if mode == GEN:
            ret_dict['sample_z'] = sample_y
            ret_dict['log_qy'] = log_qy
            return ret_dict, labels

        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            # regularization qy to be uniform
            avg_log_qy = th.exp(log_qy.view(-1, self.config.y_size, self.config.k_size))
            avg_log_qy = th.log(th.mean(avg_log_qy, dim=0) + 1e-15)
            b_pr = self.cat_kl_loss(avg_log_qy, self.log_uniform_y, batch_size, unit_average=True)
            mi = self.entropy_loss(avg_log_qy, unit_average=True) - self.entropy_loss(log_qy, unit_average=True)
            pi_kl = self.cat_kl_loss(log_qy, log_py, batch_size, unit_average=True)
            q_y = th.exp(log_qy).view(-1, self.config.y_size, self.config.k_size)  # b
            p = th.pow(th.bmm(q_y, th.transpose(q_y, 1, 2)) - self.eye, 2)

            result['pi_kl'] = pi_kl

            result['diversity'] = th.mean(p)
            result['nll'] = self.nll(dec_outputs, labels)
            result['b_pr'] = b_pr
            result['mi'] = mi
            if self.training and self.gbn:
                result['TM_loss'] = -self.LB*self.gbnrate
            return result

    def forward_rl(self, data_feed, max_words, temp=0.1):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        if self.gbn:
            topic_context = self.np2var(data_feed['sents'], LONG)
            if topic_context.size(0) < self.config.batch_size:
                topic_contextss = topic_context.reshape(topic_context.size(0),1,topic_context.size(-1))
            else:
                topic_contextss = topic_context.reshape(int(self.config.batch_size/4),-1,topic_context.size(-1))
            self.LB = self.gbn_topic.topic_driven(topic_contextss)
            enc_last = self.gbn_topic.GRU_theta_hidden(enc_last)

        # create decoder initial states
        if self.simple_posterior:
            logits_py, log_qy = self.c2z(enc_last)
        else:
            logits_py, log_qy = self.c2z(enc_last)

        qy = F.softmax(logits_py / temp, dim=1)  # (batch_size, vocab_size, )
        log_qy = F.log_softmax(logits_py, dim=1)  # (batch_size, vocab_size, )
        idx = th.multinomial(qy, 1).detach()
        logprob_sample_z = log_qy.gather(1, idx).view(-1, self.y_size)
        joint_logpz = th.sum(logprob_sample_z, dim=1)
        sample_y = cast_type(Variable(th.zeros(log_qy.size())), FLOAT, self.use_gpu)
        sample_y.scatter_(1, idx, 1.0)

        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        # decode
        logprobs, outs = self.decoder.forward_rl(batch_size=batch_size,
                                                 dec_init_state=dec_init_state,
                                                 attn_context=attn_context,
                                                 vocab=self.vocab,
                                                 max_words=max_words,
                                                 temp=0.1)
        return logprobs, outs, joint_logpz, sample_y



    def get_draw_dots(self, data_feed, mode, doc_num_batches=0, MBObserved=0, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        if self.gbn:
            self.gbn_topic.PhiPi()
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)

        # gbn
        # TODO 调整topic word context
        if self.gbn:
            topic_context = self.np2var(data_feed['sents'], LONG)
            if topic_context.size(0) < self.config.batch_size:
                topic_contextss = topic_context.reshape(topic_context.size(0),1,topic_context.size(-1))
            else:
                topic_contextss = topic_context.reshape(int(self.config.batch_size/4),-1,topic_context.size(-1))
            # self.topic_driven(topic_contextss)
            self.LB = self.gbn_topic.topic_driven(topic_contextss, doc_num_batches, MBObserved)
            # self.HT = th.cat([self.theta_1C_HT,self.theta_2C_HT,self.theta_3C_HT],0).reshape(230,-1).T
            # self.HT = th.cat([self.theta_1C_NORM,self.theta_2C_NORM,self.theta_3C_NORM],0).reshape(230,-1).T
            # self.HT = [self.theta_1C_NORM.T.reshape(batch_size,-1),self.theta_2C_NORM.T.reshape(batch_size,-1),self.theta_3C_NORM.T.reshape(batch_size,-1)]
            # DO
            enc_last = self.gbn_topic.GRU_theta_hidden(enc_last)


        self.LB = 0

        # create decoder initial states
        if self.simple_posterior:
            logits_qy, log_qy = self.c2z(enc_last)
            sample_y = self.gumbel_connector(logits_qy, hard=mode==GEN)
            log_py = self.log_uniform_y
        else:
            logits_py, log_py = self.c2z(enc_last)
            # encode response and use posterior to find q(z|x, c)
            x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1))
            if self.contextual_posterior:
                logits_qy, log_qy = self.xc2z(th.cat([enc_last, x_h.squeeze(1)], dim=1))
            else:
                logits_qy, log_qy = self.xc2z(x_h.squeeze(1))

            # use prior at inference time, otherwise use posterior
            if mode == GEN or (use_py is not None and use_py is True):
                sample_y = self.gumbel_connector(logits_py, hard=False)
            else:
                sample_y = self.gumbel_connector(logits_qy, hard=True)


        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               # (batch_size, response_size-1)
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               # (batch_size, max_ctx_len, ctx_cell_size)
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size)  # (batch_size, goal_nhid)

        ret_dict['sample_z'] = sample_y
        ret_dict['log_qy'] = log_qy
        return ret_dict, labels




class GBNSGauss(BaseModel):
    def __init__(self, corpus, config):
        super(GBNSGauss, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        self.y_size = config.y_size
        self.simple_posterior = config.simple_posterior

        self.embedding = None
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)
        self.c2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size + self.db_size + self.bs_size,
                                          config.y_size, is_lstm=False)
        self.gauss_connector = nn_lib.GaussianConnector(self.use_gpu)
        self.z_embedding = nn.Linear(self.y_size, config.dec_cell_size)
        if not self.simple_posterior:
            self.xc2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size * 2 + self.db_size + self.bs_size,
                                               config.y_size, is_lstm=False)

        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size,
                                  hidden_size=config.dec_cell_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=config.dec_cell_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.bos_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.max_dec_len,
                                  embedding=self.embedding)

        self.nll = NLLEntropy(self.pad_id, config.avg_type)
        self.gauss_kl = NormKLLoss(unit_average=True)
        self.zero = cast_type(th.zeros(1), FLOAT, self.use_gpu)

        # GBN
        self.gbn = config.use_gbn
        self.gbnrate = config.gbn_rate
        if self.gbn:
            h_size = self.utt_encoder.output_size + self.db_size + self.bs_size
            self.gbn_topic = GBNModel(config, h_size, corpus.V_tm)
        self.HT = None

    def forward(self, data_feed, mode, doc_num_batches=0, MBObserved=0, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        if self.gbn:
            self.gbn_topic.PhiPi()
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)
        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))
        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)

        # gbn
        # TODO 调整topic word context
        if self.gbn:
            topic_context = self.np2var(data_feed['sents'], LONG)
            if topic_context.size(0) < self.config.batch_size:
                topic_contextss = topic_context.reshape(topic_context.size(0),1,topic_context.size(-1))
            else:
                topic_contextss = topic_context.reshape(int(self.config.batch_size/4),-1,topic_context.size(-1))
            # self.topic_driven(topic_contextss)
            self.LB = self.gbn_topic.topic_driven(topic_contextss, doc_num_batches, MBObserved)
            enc_last = self.gbn_topic.GRU_theta_hidden(enc_last)

            if self.gbn and self.training:
                self.gbn_topic.update()
            else:
                self.LB = 0

        # create decoder initial states
        if self.simple_posterior:
            q_mu, q_logvar = self.c2z(enc_last)
            sample_z = self.gauss_connector(q_mu, q_logvar)
            p_mu, p_logvar = self.zero, self.zero
        else:
            p_mu, p_logvar = self.c2z(enc_last)
            # encode response and use posterior to find q(z|x, c)
            x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1))
            q_mu, q_logvar = self.xc2z(th.cat([enc_last, x_h.squeeze(1)], dim=1))

            # use prior at inference time, otherwise use posterior
            if mode == GEN or use_py:
                sample_z = self.gauss_connector(p_mu, p_logvar)
            else:
                sample_z = self.gauss_connector(q_mu, q_logvar)

        # pack attention context
        dec_init_state = self.z_embedding(sample_z.unsqueeze(0))
        # if self.gbn:
        #     dec_init_state = self.GRU_theta_hidden_after(dec_init_state.squeeze(),self.HT).unsqueeze(0)

        attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size
                                                               )  # (batch_size, goal_nhid)
        if mode == GEN:
            ret_dict['sample_z'] = sample_z
            return ret_dict, labels

        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            pi_kl = self.gauss_kl(q_mu, q_logvar, p_mu, p_logvar)
            result['pi_kl'] = pi_kl
            result['nll'] = self.nll(dec_outputs, labels)
            if self.training and self.gbn:
                result['TM_loss'] = -self.LB*self.gbnrate
            return result

    def gaussian_logprob(self, mu, logvar, sample_z):
        var = th.exp(logvar)
        constant = float(-0.5 * np.log(2*np.pi))
        logprob = constant - 0.5 * logvar - th.pow((mu-sample_z), 2) / (2.0*var)
        return logprob

    def forward_rl(self, data_feed, max_words, temp=0.1):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)

        if self.gbn:
            topic_context = self.np2var(data_feed['sents'], LONG)
            if topic_context.size(0) < self.config.batch_size:
                topic_contextss = topic_context.reshape(topic_context.size(0),1,topic_context.size(-1))
            else:
                topic_contextss = topic_context.reshape(int(self.config.batch_size/4),-1,topic_context.size(-1))
            self.LB = self.gbn_topic.topic_driven(topic_contextss)
            enc_last = self.gbn_topic.GRU_theta_hidden(enc_last)

        # create decoder initial states
        p_mu, p_logvar = self.c2z(enc_last)

        sample_z = th.normal(p_mu, th.sqrt(th.exp(p_logvar))).detach()
        logprob_sample_z = self.gaussian_logprob(p_mu, self.zero, sample_z)
        joint_logpz = th.sum(logprob_sample_z, dim=1)

        # pack attention context
        dec_init_state = self.z_embedding(sample_z.unsqueeze(0))
        attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        # decode
        logprobs, outs = self.decoder.forward_rl(batch_size=batch_size,
                                                 dec_init_state=dec_init_state,
                                                 attn_context=attn_context,
                                                 vocab=self.vocab,
                                                 max_words=max_words,
                                                 temp=0.1)
        return logprobs, outs, joint_logpz, sample_z



class GBNSCat_L1(BaseModel):
    def __init__(self, corpus, config):
        super(GBNSCat_L1, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        self.k_size = config.k_size
        self.y_size = config.y_size
        self.simple_posterior = config.simple_posterior
        self.contextual_posterior = config.contextual_posterior

        self.embedding = None
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)

        self.c2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size + self.db_size + self.bs_size,
                                          config.y_size, config.k_size, is_lstm=False)
        self.z_embedding = nn.Linear(self.y_size * self.k_size, config.dec_cell_size, bias=False)
        self.gumbel_connector = nn_lib.GumbelConnector(config.use_gpu)
        if not self.simple_posterior:
            if self.contextual_posterior:
                self.xc2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size * 2 + self.db_size + self.bs_size,
                                                   config.y_size, config.k_size, is_lstm=False)
            else:
                self.xc2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size, config.y_size, config.k_size, is_lstm=False)

        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size,
                                  hidden_size=config.dec_cell_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=config.dec_cell_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.bos_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.max_dec_len,
                                  embedding=self.embedding)

        self.nll = NLLEntropy(self.pad_id, config.avg_type)
        self.cat_kl_loss = CatKLLoss()
        self.entropy_loss = Entropy()
        self.log_uniform_y = Variable(th.log(th.ones(1) / config.k_size))
        self.eye = Variable(th.eye(self.config.y_size).unsqueeze(0))
        self.beta = self.config.beta if hasattr(self.config, 'beta') else 0.0
        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y.cuda()
            self.eye = self.eye.cuda()

        # GBN
        self.gbn = config.use_gbn
        self.gbnrate = config.gbn_rate
        if self.gbn:
            h_size = self.utt_encoder.output_size + self.db_size + self.bs_size
            self.gbn_topic = GBNModel_L1(config, h_size, corpus.V_tm)
        self.HT = None

    def valid_loss(self, loss, batch_cnt=None):
        if self.simple_posterior:
            if self.training and self.gbn:
                total_loss = loss.nll + loss.TM_loss
            else:
                total_loss = loss.nll
            if self.config.use_pr > 0.0:
                total_loss += self.beta * loss.pi_kl
        else:
            total_loss = loss.nll + loss.pi_kl

        if self.config.use_mi:
            total_loss += (loss.b_pr * self.beta)

        if self.config.use_diversity:
            total_loss += loss.diversity

        return total_loss

    def forward(self, data_feed, mode, doc_num_batches=0, MBObserved=0, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        if self.gbn:
            self.gbn_topic.PhiPi()
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)

        # gbn
        # TODO 调整topic word context
        if self.gbn:
            topic_context = self.np2var(data_feed['sents'], LONG)
            if topic_context.size(0) < self.config.batch_size:
                topic_contextss = topic_context.reshape(topic_context.size(0),1,topic_context.size(-1))
            else:
                topic_contextss = topic_context.reshape(int(self.config.batch_size/4),-1,topic_context.size(-1))
            # self.topic_driven(topic_contextss)
            self.LB = self.gbn_topic.topic_driven(topic_contextss, doc_num_batches, MBObserved)
            # self.HT = th.cat([self.theta_1C_HT,self.theta_2C_HT,self.theta_3C_HT],0).reshape(230,-1).T
            # self.HT = th.cat([self.theta_1C_NORM,self.theta_2C_NORM,self.theta_3C_NORM],0).reshape(230,-1).T
            # self.HT = [self.theta_1C_NORM.T.reshape(batch_size,-1),self.theta_2C_NORM.T.reshape(batch_size,-1),self.theta_3C_NORM.T.reshape(batch_size,-1)]
            # DO
            enc_last = self.gbn_topic.GRU_theta_hidden(enc_last)

        if self.gbn and self.training:
            self.gbn_topic.update()
        else:
            self.LB = 0

        # create decoder initial states
        if self.simple_posterior:
            logits_qy, log_qy = self.c2z(enc_last)
            sample_y = self.gumbel_connector(logits_qy, hard=mode==GEN)
            log_py = self.log_uniform_y
        else:
            logits_py, log_py = self.c2z(enc_last)
            # encode response and use posterior to find q(z|x, c)
            x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1))
            if self.contextual_posterior:
                logits_qy, log_qy = self.xc2z(th.cat([enc_last, x_h.squeeze(1)], dim=1))
            else:
                logits_qy, log_qy = self.xc2z(x_h.squeeze(1))

            # use prior at inference time, otherwise use posterior
            if mode == GEN or (use_py is not None and use_py is True):
                sample_y = self.gumbel_connector(logits_py, hard=False)
            else:
                sample_y = self.gumbel_connector(logits_qy, hard=True)

        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               # (batch_size, response_size-1)
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               # (batch_size, max_ctx_len, ctx_cell_size)
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size)  # (batch_size, goal_nhid)
        if mode == GEN:
            ret_dict['sample_z'] = sample_y
            ret_dict['log_qy'] = log_qy
            return ret_dict, labels

        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            # regularization qy to be uniform
            avg_log_qy = th.exp(log_qy.view(-1, self.config.y_size, self.config.k_size))
            avg_log_qy = th.log(th.mean(avg_log_qy, dim=0) + 1e-15)
            b_pr = self.cat_kl_loss(avg_log_qy, self.log_uniform_y, batch_size, unit_average=True)
            mi = self.entropy_loss(avg_log_qy, unit_average=True) - self.entropy_loss(log_qy, unit_average=True)
            pi_kl = self.cat_kl_loss(log_qy, log_py, batch_size, unit_average=True)
            q_y = th.exp(log_qy).view(-1, self.config.y_size, self.config.k_size)  # b
            p = th.pow(th.bmm(q_y, th.transpose(q_y, 1, 2)) - self.eye, 2)

            result['pi_kl'] = pi_kl

            result['diversity'] = th.mean(p)
            result['nll'] = self.nll(dec_outputs, labels)
            result['b_pr'] = b_pr
            result['mi'] = mi
            if self.training and self.gbn:
                result['TM_loss'] = -self.LB*self.gbnrate
            return result

    def forward_rl(self, data_feed, max_words, temp=0.1):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        if self.gbn:
            topic_context = self.np2var(data_feed['sents'], LONG)
            if topic_context.size(0) < self.config.batch_size:
                topic_contextss = topic_context.reshape(topic_context.size(0),1,topic_context.size(-1))
            else:
                topic_contextss = topic_context.reshape(int(self.config.batch_size/4),-1,topic_context.size(-1))
            self.LB = self.gbn_topic.topic_driven(topic_contextss)
            enc_last = self.gbn_topic.GRU_theta_hidden(enc_last)

        # create decoder initial states
        if self.simple_posterior:
            logits_py, log_qy = self.c2z(enc_last)
        else:
            logits_py, log_qy = self.c2z(enc_last)

        qy = F.softmax(logits_py / temp, dim=1)  # (batch_size, vocab_size, )
        log_qy = F.log_softmax(logits_py, dim=1)  # (batch_size, vocab_size, )
        idx = th.multinomial(qy, 1).detach()
        logprob_sample_z = log_qy.gather(1, idx).view(-1, self.y_size)
        joint_logpz = th.sum(logprob_sample_z, dim=1)
        sample_y = cast_type(Variable(th.zeros(log_qy.size())), FLOAT, self.use_gpu)
        sample_y.scatter_(1, idx, 1.0)

        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        # decode
        logprobs, outs = self.decoder.forward_rl(batch_size=batch_size,
                                                 dec_init_state=dec_init_state,
                                                 attn_context=attn_context,
                                                 vocab=self.vocab,
                                                 max_words=max_words,
                                                 temp=0.1)
        return logprobs, outs, joint_logpz, sample_y





class GBNSGauss_L1(BaseModel):
    def __init__(self, corpus, config):
        super(GBNSGauss_L1, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        self.y_size = config.y_size
        self.simple_posterior = config.simple_posterior

        self.embedding = None
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)
        self.c2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size + self.db_size + self.bs_size,
                                          config.y_size, is_lstm=False)
        self.gauss_connector = nn_lib.GaussianConnector(self.use_gpu)
        self.z_embedding = nn.Linear(self.y_size, config.dec_cell_size)
        if not self.simple_posterior:
            self.xc2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size * 2 + self.db_size + self.bs_size,
                                               config.y_size, is_lstm=False)

        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size,
                                  hidden_size=config.dec_cell_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=config.dec_cell_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.bos_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.max_dec_len,
                                  embedding=self.embedding)

        self.nll = NLLEntropy(self.pad_id, config.avg_type)
        self.gauss_kl = NormKLLoss(unit_average=True)
        self.zero = cast_type(th.zeros(1), FLOAT, self.use_gpu)

        # GBN
        self.gbn = config.use_gbn
        self.gbnrate = config.gbn_rate
        if self.gbn:
            h_size = self.utt_encoder.output_size + self.db_size + self.bs_size
            self.gbn_topic = GBNModel_L1(config, h_size, corpus.V_tm)
        self.HT = None

    def forward(self, data_feed, mode, doc_num_batches=0, MBObserved=0, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        if self.gbn:
            self.gbn_topic.PhiPi()
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)
        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))
        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)

        # gbn
        # TODO 调整topic word context
        if self.gbn:
            topic_context = self.np2var(data_feed['sents'], LONG)
            if topic_context.size(0) < self.config.batch_size:
                topic_contextss = topic_context.reshape(topic_context.size(0),1,topic_context.size(-1))
            else:
                topic_contextss = topic_context.reshape(int(self.config.batch_size/4),-1,topic_context.size(-1))
            # self.topic_driven(topic_contextss)
            self.LB = self.gbn_topic.topic_driven(topic_contextss, doc_num_batches, MBObserved)
            enc_last = self.gbn_topic.GRU_theta_hidden(enc_last)

            if self.gbn and self.training:
                self.gbn_topic.update()
            else:
                self.LB = 0

        # create decoder initial states
        if self.simple_posterior:
            q_mu, q_logvar = self.c2z(enc_last)
            sample_z = self.gauss_connector(q_mu, q_logvar)
            p_mu, p_logvar = self.zero, self.zero
        else:
            p_mu, p_logvar = self.c2z(enc_last)
            # encode response and use posterior to find q(z|x, c)
            x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1))
            q_mu, q_logvar = self.xc2z(th.cat([enc_last, x_h.squeeze(1)], dim=1))

            # use prior at inference time, otherwise use posterior
            if mode == GEN or use_py:
                sample_z = self.gauss_connector(p_mu, p_logvar)
            else:
                sample_z = self.gauss_connector(q_mu, q_logvar)

        # pack attention context
        dec_init_state = self.z_embedding(sample_z.unsqueeze(0))
        # if self.gbn:
        #     dec_init_state = self.GRU_theta_hidden_after(dec_init_state.squeeze(),self.HT).unsqueeze(0)

        attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size
                                                               )  # (batch_size, goal_nhid)
        if mode == GEN:
            ret_dict['sample_z'] = sample_z
            return ret_dict, labels

        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            pi_kl = self.gauss_kl(q_mu, q_logvar, p_mu, p_logvar)
            result['pi_kl'] = pi_kl
            result['nll'] = self.nll(dec_outputs, labels)
            if self.training and self.gbn:
                result['TM_loss'] = -self.LB*self.gbnrate
            return result

    def gaussian_logprob(self, mu, logvar, sample_z):
        var = th.exp(logvar)
        constant = float(-0.5 * np.log(2*np.pi))
        logprob = constant - 0.5 * logvar - th.pow((mu-sample_z), 2) / (2.0*var)
        return logprob

    def forward_rl(self, data_feed, max_words, temp=0.1):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)

        if self.gbn:
            topic_context = self.np2var(data_feed['sents'], LONG)
            if topic_context.size(0) < self.config.batch_size:
                topic_contextss = topic_context.reshape(topic_context.size(0),1,topic_context.size(-1))
            else:
                topic_contextss = topic_context.reshape(int(self.config.batch_size/4),-1,topic_context.size(-1))
            self.LB = self.gbn_topic.topic_driven(topic_contextss)
            enc_last = self.gbn_topic.GRU_theta_hidden(enc_last)

        # create decoder initial states
        p_mu, p_logvar = self.c2z(enc_last)

        sample_z = th.normal(p_mu, th.sqrt(th.exp(p_logvar))).detach()
        logprob_sample_z = self.gaussian_logprob(p_mu, self.zero, sample_z)
        joint_logpz = th.sum(logprob_sample_z, dim=1)

        # pack attention context
        dec_init_state = self.z_embedding(sample_z.unsqueeze(0))
        attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        # decode
        logprobs, outs = self.decoder.forward_rl(batch_size=batch_size,
                                                 dec_init_state=dec_init_state,
                                                 attn_context=attn_context,
                                                 vocab=self.vocab,
                                                 max_words=max_words,
                                                 temp=0.1)
        return logprobs, outs, joint_logpz, sample_z


