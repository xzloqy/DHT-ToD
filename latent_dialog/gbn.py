import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from latent_dialog.utils import INT, FLOAT, LONG, cast_type
from gbn_woz.utils import initialize_Phi_Pi, update_Pi_Phi, Bow_sents


class GBNModel(nn.Module):
    def __init__(self, config, h_size, V_tm):
        super(GBNModel, self).__init__()
        # self.use_gpu = config.use_gpu
        self.config = config
        self.gbn = config.use_gbn
        self.gbnrate = config.gbn_rate
        # self.K_dim = (100,80,50)
        # self.K_dim = (50,30,20)
        self.real_min = 2.2e-20

        self.Phi, self.Pi, self.NDot_Phi, self.NDot_Pi = initialize_Phi_Pi(V_tm)
        self.K_dim = (self.Phi[0].shape[1],self.Phi[1].shape[1],self.Phi[2].shape[1])
        self.h2zr = nn.Linear(self.K_dim[0]+h_size,h_size*2)
        self.t2c = nn.Linear(self.K_dim[0],h_size,bias=False)
        self.h2c = nn.Linear(h_size,h_size)
        self.h2zr2 = nn.Linear(self.K_dim[1]+h_size,h_size*2)
        self.t2c2 = nn.Linear(self.K_dim[1],h_size,bias=False)
        self.h2c2 = nn.Linear(h_size,h_size)
        self.h2zr3 = nn.Linear(self.K_dim[2]+h_size,h_size*2)
        self.t2c3 = nn.Linear(self.K_dim[2],h_size,bias=False)
        self.h2c3 = nn.Linear(h_size,h_size)

        self.state1 = nn.Sequential(nn.Linear(V_tm, self.K_dim[0], bias=False),
                                    nn.Sigmoid(),)
        self.state2 = nn.Sequential(nn.Linear(self.K_dim[0], self.K_dim[1], bias=False),
                                    nn.Sigmoid(),)
        self.state3 = nn.Sequential(nn.Linear(self.K_dim[1], self.K_dim[2], bias=False),
                                    nn.Sigmoid(),)

        self.Weilbullk = nn.ModuleList()
        self.Weilbullk.append(nn.Sequential(
                        nn.Linear(self.K_dim[0],1),
                        nn.Softplus()))
        self.Weilbullk.append(nn.Sequential(
                        nn.Linear(self.K_dim[1],1),
                        nn.Softplus()))
        self.Weilbullk.append(nn.Sequential(
                        nn.Linear(self.K_dim[2],1),
                        nn.Softplus()))
        self.Weilbulll = nn.ModuleList()
        self.Weilbulll.append(nn.Sequential(
                        nn.Linear(self.K_dim[0],self.K_dim[0]),
                        nn.Softplus()))
        self.Weilbulll.append(nn.Sequential(
                        nn.Linear(self.K_dim[1],self.K_dim[1]),
                        nn.Softplus()))
        self.Weilbulll.append(nn.Sequential(
                        nn.Linear(self.K_dim[2],self.K_dim[2]),
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

    def topic_driven(self, topic_context, doc_num_batches=0, MBObserved=0):
        # context = th.tensor(topic_context).cuda()
        context = topic_context
        self.topic_context = topic_context
        self.doc_num_batches = doc_num_batches
        self.MBObserved = MBObserved
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
            theta_1c_norm = theta_1c / th.clamp(theta_1c.max(1)[0].reshape(batch_size,1).repeat(1,theta_1c.size(1)), min=self.real_min)
            theta_2c_norm = theta_2c / th.clamp(theta_2c.max(1)[0].reshape(batch_size,1).repeat(1,theta_2c.size(1)), min=self.real_min)
            theta_3c_norm = theta_3c / th.clamp(theta_3c.max(1)[0].reshape(batch_size,1).repeat(1,theta_3c.size(1)), min=self.real_min)
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
        batch_size = batch_size*sent_J
        self.HT = [self.theta_1C_NORM.T.reshape(batch_size,-1),self.theta_2C_NORM.T.reshape(batch_size,-1),self.theta_3C_NORM.T.reshape(batch_size,-1)]
        return self.LB

    def GRU_theta_hidden(self,hidden):
        theta = self.HT
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

    def update(self):
        self.Phi, self.Pi, self.NDot_Phi, self.NDot_Pi = update_Pi_Phi(self.topic_context, self.Phi, self.Pi, [self.theta_1C_HT,self.theta_2C_HT,self.theta_3C_HT], self.doc_num_batches, self.MBObserved, self.NDot_Phi, self.NDot_Pi)
    


