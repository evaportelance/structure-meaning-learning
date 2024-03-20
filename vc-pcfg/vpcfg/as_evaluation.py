import time
import numpy as np
from collections import OrderedDict
import torch
from . import utils
import torch.nn as nn

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)

class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s


def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    model.eval()
    end = time.time()
    
    n_word, n_sent = 0, 0
    total_ll, total_kl = 0., 0.
    sent_f1, corpus_f1 = [], [0., 0., 0.] 

    img_embs = None
    cap_embs = None
    for i, (images, captions, lengths, ids, spans) in enumerate(data_loader):
        model.logger = val_logger
        lengths = torch.tensor(lengths).long() if isinstance(lengths, list) else lengths

        bsize = captions.size(0) 
        img_emb, cap_span_features, nll, kl, span_margs, argmax_spans, trees, lprobs = \
            model.forward_encoder(
                images, captions, lengths, spans, require_grad=False
            )
        mstep = (lengths * (lengths - 1) / 2).int() # (b, NT, dim) 
        cap_feats = torch.cat(
            [cap_span_features[j][k - 1].unsqueeze(0) for j, k in enumerate(mstep)], dim=0
        ) 
        span_marg = torch.softmax(
            torch.cat([span_margs[j][k - 1].unsqueeze(0) for j, k in enumerate(mstep)], dim=0), -1
        )
        cap_emb = torch.bmm(span_marg.unsqueeze(-2),  cap_feats).squeeze(-2)
        cap_emb = utils.l2norm(cap_emb)

        # initialize the numpy arrays given the size of the embeddings
        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))

        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids] = cap_emb.data.cpu().numpy().copy()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        total_ll += nll.sum().item()
        total_kl += kl.sum().item()
        n_word += (lengths + 1).sum().item()
        n_sent += bsize

        bsize = img_emb.shape[0]
        for b in range(bsize):
            max_len = lengths[b].item() 
            pred = [(a[0], a[1]) for a in argmax_spans[b] if a[0] != a[1]]
            pred_set = set(pred[:-1])
            gold = [(spans[b][i][0].item(), spans[b][i][1].item()) for i in range(max_len - 1)] 
            gold_set = set(gold[:-1])

            tp, fp, fn = utils.get_stats(pred_set, gold_set) 
            corpus_f1[0] += tp
            corpus_f1[1] += fp
            corpus_f1[2] += fn
            
            overlap = pred_set.intersection(gold_set)
            prec = float(len(overlap)) / (len(pred_set) + 1e-8)
            reca = float(len(overlap)) / (len(gold_set) + 1e-8)
            
            if len(gold_set) == 0:
                reca = 1. 
                if len(pred_set) == 0:
                    prec = 1.
            f1 = 2 * prec * reca / (prec + reca + 1e-8)
            sent_f1.append(f1)

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                        i, len(data_loader), batch_time=batch_time,
                        e_log=str(model.logger)))
        del images, captions
        #if i >= 50: break

    tp, fp, fn = corpus_f1  
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    corpus_f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.
    sent_f1 = np.mean(np.array(sent_f1))
    recon_ppl = np.exp(total_ll / n_word)
    ppl_elbo = np.exp((total_ll + total_kl) / n_word) 
    kl = total_kl / n_sent
    info = '\nReconPPL: {:.2f}, KL: {:.4f}, PPL (Upper Bound): {:.2f}\n' + \
           'Corpus F1: {:.2f}, Sentence F1: {:.2f}'
    info = info.format(
        recon_ppl, kl, ppl_elbo, corpus_f1 * 100, sent_f1 * 100
    )
    logging(info)
    return img_embs, cap_embs, ppl_elbo, sent_f1 * 100 

def i2t(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = int(images.shape[0] / 5)
        # print(npts)
    index_list = []

    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[5 * index].reshape(1, images.shape[1])

        # Compute scores
        d = np.dot(im, captions.T).flatten()
        inds = np.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = int(images.shape[0] / 5)
    ims = np.array([images[i] for i in range(0, len(images), 5)])

    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[5 * index:5 * index + 5]

        # compute scores
        d = np.dot(queries, ims.T)
        inds = np.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = np.argsort(d[i])[::-1]
            ranks[5 * index + i] = np.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    # compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)

def validate(opt, val_loader, model, logger):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs, val_ppl, val_f1 = encode_data(
        model, val_loader, opt.log_step, logger.info)
    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, measure='cosine')
    logger.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))
    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(
        img_embs, cap_embs, measure='cosine')
    logger.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanr))
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i
    return val_ppl

def semantic_bootstrapping_test(opt, sem_data_loader, model, logger, current_epoch, save=True):
    #if visual_mode:
    #    return validate(opt, data_loader, model, logger)
    batch_time = AverageMeter()
    val_logger = LogCollector()
    model.eval()
    end = time.time()
    nbatch = len(sem_data_loader)

    n_word, n_sent = 0, 0
    sent_f1, corpus_f1 = [], [0., 0., 0.] 
    total_ll, total_kl = 0., 0.
    ids_all = []
    pred_spans = []
    gold_spans = []
    for i, (images, captions, lengths, ids, spans) in enumerate(sem_data_loader):
        model.logger = val_logger
        if torch.cuda.is_available():
            if isinstance(lengths, list):
                lengths = torch.tensor(lengths).long()
            lengths = lengths.cuda()
            captions = captions.cuda()
        bsize = captions.size(0) 
        nll, kl, span_margs, argmax_spans, trees, lprobs = model.forward_parser(captions, lengths)
        batch_time.update(time.time() - end)
        end = time.time()
        total_ll += nll.sum().item()
        total_kl += kl.sum().item()
        n_word += (lengths + 1).sum().item()
        n_sent += bsize
        for b in range(bsize):           
            max_len = lengths[b].item() 
            pred = [(a[0], a[1]) for a in argmax_spans[b] if a[0] != a[1]]
            pred_set = set(pred[:-1])
            gold = [(spans[b][i][0].item(), spans[b][i][1].item()) for i in range(max_len - 1)] 
            gold_set = set(gold[:-1])
            # scores are calculated on inside branching, excluding terminal branches and final root branch
            tp, fp, fn = utils.get_stats(pred_set, gold_set) 
            corpus_f1[0] += tp
            corpus_f1[1] += fp
            corpus_f1[2] += fn          
            overlap = pred_set.intersection(gold_set)
            prec = float(len(overlap)) / (len(pred_set) + 1e-8)
            reca = float(len(overlap)) / (len(gold_set) + 1e-8)           
            if len(gold_set) == 0:
                reca = 1. 
                if len(pred_set) == 0:
                    prec = 1.
            f1 = 2 * prec * reca / (prec + reca + 1e-8)
            sent_f1.append(f1)
            ids_all.append(ids[b])
            pred_spans.append(pred)
            gold_spans.append(gold)
        if i % model.log_step == 0:
            logger.info(
                'Test: [{0}/{1}]\t{e_log}\t'
                .format(
                    i, nbatch, e_log=str(model.logger)
                )
            )
        del captions, lengths, ids, spans
        #if i > 10: break
    tp, fp, fn = corpus_f1  
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    corpus_f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.
    mean_sent_f1 = np.mean(np.array(sent_f1))
    recon_ppl = np.exp(total_ll / n_word)
    ppl_elbo = np.exp((total_ll + total_kl) / n_word) 
    kl = total_kl / n_sent
    info = '\nReconPPL: {:.2f}, KL: {:.4f}, PPL (Upper Bound): {:.2f}\n' + \
           'Corpus F1: {:.2f}, Sentence F1: {:.2f}'
    info = info.format(
        recon_ppl, kl, ppl_elbo, corpus_f1 * 100, mean_sent_f1 * 100
    )
    logger.info(info)
    if save:
        file = opt.logger_name + '/semantic_bootstrapping_results/' + str(current_epoch) +'.csv'
        utils.save_columns_to_csv(file, ids_all, gold_spans, pred_spans, sent_f1)
    return ppl_elbo 


def syntactic_bootstrapping_test(opt, syn_data_loader, model, logger, current_epoch, save=True):
    batch_time = AverageMeter()
    val_logger = LogCollector()
    sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    model.eval()
    end = time.time()
    ids_transitives = []
    ids_intransitives = []
    cap_score_tr = []
    cap_score_intr = []
    tree_score_tr = []
    tree_score_intr = []
    span_score_tr = []
    span_score_intr = []
    short_tree_score_tr = []
    short_tree_score_intr = []
    short_span_score_tr = []
    short_span_score_intr = []
#     itr_ctr_all = []
#     itr_cintr_all = []
#     iintr_cintr_all = []
#     iintr_ctr_all = []
    for i, (images_tr, captions_tr, lengths_tr, ids_tr, spans_tr, images_intr, captions_intr, lengths_intr, ids_intr, spans_intr) in enumerate(syn_data_loader): 
        if isinstance(lengths_tr, list):
            lengths_tr = torch.tensor(lengths_tr).long()
            lengths_intr = torch.tensor(lengths_intr).long()      
        bsize = captions_tr.size(0)
        images = torch.cat((images_tr, images_intr, images_intr, images_tr), 0)
        captions = torch.cat((captions_tr, captions_tr, captions_intr, captions_intr), 0)
        lengths = torch.cat((lengths_tr, lengths_tr, lengths_intr, lengths_intr), 0)
        spans = torch.cat((spans_tr, spans_tr, spans_intr, spans_intr), 0)
        if torch.cuda.is_available():
            lengths = lengths.cuda()
            captions = captions.cuda()
            images = images.cuda()
        model.logger = val_logger
        img_emb, cap_span_features, nll, kl, span_margs, argmax_spans, trees, lprobs = \
            model.forward_encoder(images, captions, lengths, spans, require_grad=False) 
        mstep = (lengths * (lengths - 1) / 2).int() # (b, NT, dim) 
        # get caption embeddings
        # if only using whole string embedding
        cap_feats = torch.cat([cap_span_features[j][k - 1].unsqueeze(0) for j, k in enumerate(mstep)], dim=0) 
        span_marg = torch.softmax(torch.cat([span_margs[j][k - 1].unsqueeze(0) for j, k in enumerate(mstep)], dim=0), -1)
        cap_emb = torch.bmm(span_marg.unsqueeze(-2),  cap_feats).squeeze(-2)
        #cap_emb = cap_feats.sum(-2)
        cap_emb = utils.l2norm(cap_emb)
        # if averaging across constituents
        tree_emb = []
        mean_span_sim = []
        short_tree_emb = []
        short_mean_span_sim = []
        for b, nstep in enumerate(mstep):
            span_embs = []
            span_sims = []
            for k in range(nstep):
                span_feats = cap_span_features[b, k] 
                span_marg = span_margs[b, k].softmax(-1).unsqueeze(-2)
                span_emb = torch.matmul(span_marg, span_feats).squeeze(-2)
                span_emb = utils.l2norm(span_emb)
                span_embs.append(span_emb)
                sim = utils.cosine_sim(img_emb[b].unsqueeze(0), span_emb.unsqueeze(0)).squeeze(0)
                span_sims.append(sim)
            #average embeddings across spans  (I could also try matching span to image directly and taking average score OR comparing spans directly... but lengths dont always match)
            ## short spans only
            nstep = int(mstep.float().mean().item() / 2)
            short_span_emb = torch.mean(torch.stack(span_embs[:nstep]), 0)
            short_span_sim = torch.mean(torch.stack(span_sims[:nstep]), 0)
            ##
            span_embs = torch.stack(span_embs)
            span_sims = torch.stack(span_sims)
            span_emb = torch.mean(span_embs, 0)
            span_sim = torch.mean(span_sims, 0)
            tree_emb.append(span_emb)
            mean_span_sim.append(span_sim)
            short_tree_emb.append(short_span_emb)
            short_mean_span_sim.append(short_span_sim)
        tree_emb = torch.stack(tree_emb)
        short_tree_emb = torch.stack(short_tree_emb) 
        # compare cosine similarity with images
        mean_span_sim = torch.cat(mean_span_sim, 0)
        short_mean_span_sim = torch.cat(short_mean_span_sim, 0)
        cap_sim = utils.cosine_sim(img_emb, cap_emb).diag()
        tree_sim =utils.cosine_sim(img_emb, tree_emb).diag()
        short_tree_sim =utils.cosine_sim(img_emb, short_tree_emb).diag()
        # split results into 4 comparison groups
        sim_tr_pos, sim_tr_neg, sim_intr_pos, sim_intr_neg = torch.split(cap_sim, bsize)
        cap_ans_tr = torch.gt(sim_tr_pos, sim_tr_neg)
        cap_ans_intr = torch.gt(sim_intr_pos, sim_intr_neg)
        sim_tr_pos, sim_tr_neg, sim_intr_pos, sim_intr_neg = torch.split(mean_span_sim, bsize)  
        span_ans_tr = torch.gt(sim_tr_pos, sim_tr_neg)
        span_ans_intr = torch.gt(sim_intr_pos, sim_intr_neg)
        sim_tr_pos, sim_tr_neg, sim_intr_pos, sim_intr_neg = torch.split(short_mean_span_sim, bsize)   
        short_span_ans_tr = torch.gt(sim_tr_pos, sim_tr_neg)
        short_span_ans_intr = torch.gt(sim_intr_pos, sim_intr_neg)
        sim_tr_pos, sim_tr_neg, sim_intr_pos, sim_intr_neg = torch.split(tree_sim, bsize) 
        tree_ans_tr = torch.gt(sim_tr_pos, sim_tr_neg)
        tree_ans_intr = torch.gt(sim_intr_pos, sim_intr_neg)
        sim_tr_pos, sim_tr_neg, sim_intr_pos, sim_intr_neg = torch.split(short_tree_sim, bsize) 
        short_tree_ans_tr = torch.gt(sim_tr_pos, sim_tr_neg)
        short_tree_ans_intr = torch.gt(sim_intr_pos, sim_intr_neg)
        ids_transitives += ids_tr
        ids_intransitives += ids_intr
        cap_score_tr += cap_ans_tr.tolist()
        cap_score_intr += cap_ans_intr.tolist()
        tree_score_tr += tree_ans_tr.tolist()
        tree_score_intr += tree_ans_intr.tolist()
        span_score_tr += span_ans_tr.tolist()
        span_score_intr += span_ans_intr.tolist()
        short_tree_score_tr += short_tree_ans_tr.tolist()
        short_tree_score_intr += short_tree_ans_intr.tolist()
        short_span_score_tr += short_span_ans_tr.tolist()
        short_span_score_intr += short_span_ans_intr.tolist()
#         itr_ctr_all += sims_tr_tr.tolist()
#         itr_cintr_all += sims_tr_intr.tolist()
#         iintr_cintr_all += sims_intr_intr.tolist()
#         iintr_ctr_all += sims_intr_tr.tolist()
        del images, captions, lengths, spans
    if save:
        file = opt.logger_name + '/syntactic_bootstrapping_results/' + str(current_epoch) +'.csv'
        utils.save_columns_to_csv(file, ids_transitives, ids_intransitives, cap_score_tr, cap_score_intr, tree_score_tr, tree_score_intr, span_score_tr, span_score_intr, short_tree_score_tr, short_tree_score_intr, short_span_score_tr, short_span_score_intr)
    n = len(ids_transitives)
    cap_score_tr = sum(cap_score_tr) / n
    cap_score_intr = sum(cap_score_intr) / n
    tree_score_tr = sum(tree_score_tr) / n
    tree_score_intr = sum(tree_score_intr) / n
    span_score_tr = sum(span_score_tr) / n
    span_score_intr = sum(span_score_intr) / n
    short_tree_score_tr = sum(short_tree_score_tr) / n
    short_tree_score_intr = sum(short_tree_score_intr) / n
    short_span_score_tr = sum(short_span_score_tr) / n
    short_span_score_intr = sum(short_span_score_intr) / n
    cap_score = (cap_score_tr + cap_score_intr) / 2
    tree_score = (tree_score_tr + tree_score_intr) / 2
    span_score = (span_score_tr + span_score_intr) / 2
    short_tree_score = (short_tree_score_tr + short_tree_score_intr) / 2
    short_span_score = (short_span_score_tr + short_span_score_intr) / 2
    info = '\nSyntactic bootstapping\n Cap score: {:.4f}, Cap transitive score: {:.4f}, Cap intransitive score: {:.4f}\n Tree score: {:.4f}, Tree transitive score: {:.4f}, Tree intransitive score: {:.4f}\n Span score: {:.4f}, Span transitive score: {:.4f}, Span intransitive score: {:.4f}\n Short Tree score: {:.4f}, Short Tree transitive score: {:.4f}, Short Tree intransitive score: {:.4f}\n Short Span score: {:.4f}, Short Span transitive score: {:.4f}, Short Span intransitive score: {:.4f}\n'
    info = info.format(cap_score, cap_score_tr, cap_score_intr, tree_score, tree_score_tr, tree_score_intr, span_score, span_score_tr, span_score_intr, short_tree_score, short_tree_score_tr, short_tree_score_intr, short_span_score, short_span_score_tr, short_span_score_intr)
    logger.info(info)
    return span_score


# img_emb_tr, img_emb_intr = torch.split(img_emb, bsize)
#         print(img_emb_tr.shape)
#         cap_span_features_tr, cap_span_features_intr = torch.split(cap_span_features, bsize)
#         print(cap_span_features_tr.shape)
#         span_margs_tr, span_margs_tr = 
#         argmax_spans_tr, argmax_spans_intr = torch.split(argmax_spans, bsize)
        
#         for b in range(bsize):
#             pred_tr = [(a[0], a[1]) for a in argmax_spans_tr[b] if a[0] != a[1]]
#             pred_intr = [(a[0], a[1]) for a in argmax_spans_intr[b] if a[0] != a[1]]
            
        
#         N = lengths.max(0)[0]
#         #nstep = int(N * (N - 1) / 2)
#         mstep = (lengths * (lengths - 1) / 2).int()
#         # focus on only short spans
#         #nstep = int(mstep.float().mean().item() / 2)
#         nstep = int(mstep.float().mean().item())
#         matching_matrix = torch.zeros(bsize, nstep, device=img_emb.device)
#         #if using constituents
#         for b in range(bsize):
#             for k in range(nstep):
#                 print(k)
#                 img_tr = img_emb_tr[b]
#                 img_intr = img_emb_intr[b]
#                 cap_tr = cap_span_features_tr[b, k]
#                 cap_intr = cap_span_features_intr[b, k] 
#                 cap_marg = span_margs[b, k].softmax(-1).unsqueeze(-2)
#                 cap_emb = torch.matmul(cap_marg, cap_emb).squeeze(-2)
#                 cap_emb = utils.l2norm(cap_emb)
                
#             print(cap_emb.shape)
#             sims = utils.cosine_sim(img_emb, cap_emb)
#             print(sims.shape)
#             print(sims)
#             break
#         break
            