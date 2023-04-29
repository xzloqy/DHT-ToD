import gbn_woz.config as cf
import numpy as np
import re
import numpy as np
from collections import defaultdict
import operator
import sys
from gbn_woz.rGBN_sampler import MyThread
from gbn_woz import GBN_sampler


def gen_vocab(dummy_symbols, vocab_dict, stopwords, ignore_delx):
    idxvocab = []
    # vocabxid = defaultdict(int)
    # vocab_freq = defaultdict(int)
    # for line_id, line in enumerate(codecs.open(corpus, "r", "utf-8")):
    #     for word in line.strip().split():
    #         vocab_freq[word] += 1
    #     if line_id % 1000 == 0 and verbose:
    #         sys.stdout.write(str(line_id) + " processed\r")
    #         sys.stdout.flush()

    # for s in dummy_symbols:
    #     update_vocab(s, idxvocab, vocabxid)


    # for w, f in sorted(vocab_freq.items(), key=operator.itemgetter(1), reverse=True):

    #     if f < vocab_minfreq:
    #         break
    #     else:
    #         update_vocab(w, idxvocab, vocabxid)

    vocabxid = vocab_dict
    stopwords = set([item.strip().lower() for item in open(stopwords)])
    alpha_check = re.compile("[a-zA-Z]")
    if ignore_delx:
        symbols = set([ w for w in vocabxid.keys() if ((alpha_check.search(w) == None) or w.startswith("[")) ])
    else:
        symbols = set([ w for w in vocabxid.keys() if (alpha_check.search(w) == None) ])
    ignore = stopwords | symbols | set(dummy_symbols) | set(["n't"])    
    ignore = set([vocabxid[w] for w in ignore if w in vocabxid])

    return ignore


def initialize_Phi_Pi(V_tm):
    Phi = [0] * cf.DPGDS_Layer
    Pi = [0] * cf.DPGDS_Layer
    NDot_Phi = [0] * cf.DPGDS_Layer

    NDot_Pi = [0] * cf.DPGDS_Layer
    for l in range(cf.DPGDS_Layer):
        if l == 0:
            Phi[l] = np.random.rand(V_tm, cf.K[l])
        else:
            Phi[l] = np.random.rand(cf.K[l - 1], cf.K[l])

        Phi[l] = Phi[l] / np.sum(Phi[l], axis=0)
        Pi[l] = np.eye(cf.K[l])
    return Phi, Pi, NDot_Phi, NDot_Pi


def update_Pi_Phi(miniBatch, Phi, Pi, Theta, MBratio, MBObserved, NDot_Phi, NDot_Pi):

    # miniBatch = miniBatch.reshape(8,-1,miniBatch.size(-1))
    ForgetRate = np.power((cf.Setting['tao0FR'] + np.linspace(1, int(cf.Setting['Iterall']), int(cf.Setting['Iterall']))),
                          -cf.Setting['kappa0FR'])
    epsit = np.power((cf.Setting['tao0'] + np.linspace(1, int(cf.Setting['Iterall']), int(cf.Setting['Iterall']))), -cf.Setting['kappa0'])
    epsit = cf.Setting['epsi0'] * epsit / epsit[0]

    L = cf.DPGDS_Layer
    A_VK = [0]* L
    L_KK = [0]* L
    Piprior = [0]* L
    EWSZS_Phi = [0]* L
    EWSZS_Pi = [0]* L

    Xi = []
    Vk = []
    for l in range(L):
        Xi.append(1)
        Vk.append(np.ones((cf.K[l], 1)))

    threads = []
    batch_size = Theta[0].size(1)
    sent_J = Theta[0].size(2)
    for i in range(batch_size):
        # Theta1 = np.transpose(Theta[0][ 4*i:4*(i+1), :].cpu().detach().numpy())
        # Theta2 = np.transpose(Theta[1][ 4*i:4*(i+1), :].cpu().detach().numpy())
        # Theta3 = np.transpose(Theta[2][ 4*i:4*(i+1), :].cpu().detach().numpy())
        Theta1 = Theta[0][ :, i, :].cpu().detach().numpy()
        Theta2 = Theta[1][ :, i, :].cpu().detach().numpy()
        Theta3 = Theta[2][ :, i, :].cpu().detach().numpy()
        batch = np.transpose(miniBatch[i, :, :].cpu())
        t = MyThread(i, batch, Phi, Theta1, Theta2, Theta3, L, cf.K , sent_J, Pi)
        threads.append(t)
    for t in threads:
        t.setDaemon(True)
        t.start()
    for t in threads:
        t.join()
    for t in threads:
        AA, BB, CC = t.get_result()
        for l in range(L):
            A_VK[l] = A_VK[l] + BB[l]
            L_KK[l] = L_KK[l] + CC[l]

    for l in range(len(Phi)):
        EWSZS_Phi[l] = MBratio * A_VK[l]
        EWSZS_Pi[l] = MBratio * L_KK[l]

        if (MBObserved == 0):
            NDot_Phi[l] = EWSZS_Phi[l].sum(0)
            NDot_Pi[l] = EWSZS_Pi[l].sum(0)
        else:
            NDot_Phi[l] = (1 - ForgetRate[MBObserved]) * NDot_Phi[l] + ForgetRate[MBObserved] * EWSZS_Phi[l].sum(
                0) 
            NDot_Pi[l] = (1 - ForgetRate[MBObserved]) * NDot_Pi[l] + ForgetRate[MBObserved] * EWSZS_Pi[l].sum(0)  

        tmp = EWSZS_Phi[l] + cf.eta0  
        tmp = (1 / np.maximum(NDot_Phi[l], cf.real_min)) * (tmp - tmp.sum(0) * Phi[l])  
        tmp1 = (2 / np.maximum(NDot_Phi[l], cf.real_min)) * Phi[l]
        tmp = Phi[l] + epsit[MBObserved] * tmp + np.sqrt(epsit[MBObserved] * tmp1) * np.random.randn(Phi[l].shape[0],Phi[l].shape[1])
        Phi[l] = GBN_sampler.ProjSimplexSpecial(tmp, Phi[l], 0)

        Piprior[l] = np.dot(Vk[l], np.transpose(Vk[l]))
        Piprior[l][np.arange(Piprior[l].shape[0]), np.arange(Piprior[l].shape[1])] = 0
        Piprior[l] = Piprior[l] + np.diag((Xi[l] * Vk[l]).reshape(Vk[l].shape[0], 1))

        tmp = EWSZS_Pi[l] + Piprior[l]  
        tmp = (1 / np.maximum(NDot_Pi[l], cf.real_min)) * (tmp - tmp.sum(0) * Pi[l])  
        tmp1 = (2 / np.maximum(NDot_Pi[l], cf.real_min)) * Pi[l]
        tmp = Pi[l] + epsit[MBObserved] * tmp + np.sqrt(epsit[MBObserved] * tmp1) * np.random.randn(Pi[l].shape[0],
                                                                                                    Pi[l].shape[1])
        Pi[l] = GBN_sampler.ProjSimplexSpecial(tmp, Pi[l], 0)

    return Phi, Pi, NDot_Phi, NDot_Pi


def pad(lst, max_len, pad_symbol):
    return lst + [pad_symbol] * (max_len - len(lst))


def Bag_of_words(TM_original_train,idxvocab,tm_ignore):  
    TM_doc = []
    for d in range(len(TM_original_train)):
        docw = TM_original_train[d]["tm_sent"]
        docw = pad([item for item in docw][:50], 50, 0)
        TM_doc.append(docw)
    TM_train_bow = np.zeros([len(idxvocab), len(TM_doc)])
    for doc_index in range(len(TM_doc)):
        for word in TM_doc[doc_index]:
            TM_train_bow[word][doc_index] += 1
    TM_train_bow = np.delete(TM_train_bow, list(tm_ignore), axis = 0)
    return TM_doc, TM_train_bow


def get_doc_sent(sents, docbow, sent_num, sent_J, pad_id=0):
    D = []
    docs = []
    for id_doc in range(len(sents)):
        doc = sents[id_doc].dlg
        adds = len(doc)
        J = int(np.ceil(float(adds) / sent_J ))
        for s_id in range(adds):
            docs.append((id_doc, s_id, doc[s_id].utt))
        for s_id in range(J * sent_J - adds):
            docs.append((id_doc, s_id + adds, [pad_id] * cf.lm_sent_len))  ######## pading 0 of sentence to the doc which is less than 8 sentences
        for id_sent in range(J):
            sent = docs[id_sent * sent_J : (id_sent+1)*sent_J]
            D.append(sent)
    return D


def Bow_sents(s,V,tm_ignore):
    s_b = np.zeros(V)
    for w in s:
        s_b[w] += 1
    s_b = np.delete(s_b, list(tm_ignore), axis=0)
    return s_b


def get_batch_tm_lm_all(Doc, docbow, V, tm_ignore, Batch_Size,  pad_id=0):
    x, y, d, m = [], [], [], []
    for idx_sent in range(Batch_Size):
        xx, yy, dd, mm = [], [], [], []
        for docid, seqid, seq in Doc[idx_sent]:
            xx.append(pad(seq[:-1], cf.lm_sent_len, pad_id))
            yy.append(pad(seq[1:], cf.lm_sent_len, pad_id))
            dd.append(docbow[: ,docid] - Bow_sents(seq,V,tm_ignore))
            if seq[0] == pad_id:
                mm.append([0.0] * cf.lm_sent_len)
            else:
                mm.append([1.0] * (len(seq) - 1) + [0.0] * (cf.lm_sent_len - len(seq) + 1))
        x.append(xx)
        y.append(yy)
        d.append(dd)
        m.append(mm)
    return  np.array(x), np.array(y), np.array(d), np.array(m)