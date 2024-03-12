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
    ans_transitives = []
    ans_intransitives = []
    itr_ctr_all = []
    itr_cintr_all = []
    iintr_cintr_all = []
    iintr_ctr_all = []
    for i, (images_tr, captions_tr, lengths_tr, ids_tr, spans_tr, images_intr, captions_intr, lengths_intr, ids_intr, spans_intr) in enumerate(syn_data_loader):       
        if isinstance(lengths_tr, list):
            lengths_tr = torch.tensor(lengths_tr).long()
            lengths_intr = torch.tensor(lengths_intr).long()      
        bsize = captions_tr.size(0)
        images = torch.cat((images_tr, images_intr), 0)
        captions = torch.cat((captions_tr, captions_intr), 0)
        lengths = torch.cat((lengths_tr, lengths_intr), 0)
        spans = torch.cat((spans_tr, spans_intr), 0)
        if torch.cuda.is_available():
            lengths = lengths.cuda()
            captions = captions.cuda()
            images = images.cuda()
        model.logger = val_logger
        img_emb, cap_span_features, nll, kl, span_margs, argmax_spans, trees, lprobs = \
            model.forward_encoder(images, captions, lengths, spans, require_grad=False)
        mstep = (lengths * (lengths - 1) / 2).int() # (b, NT, dim) 
        # get caption embeddings
        cap_feats = torch.cat([cap_span_features[j][k - 1].unsqueeze(0) for j, k in enumerate(mstep)], dim=0) 
        span_marg = torch.softmax(torch.cat([span_margs[j][k - 1].unsqueeze(0) for j, k in enumerate(mstep)], dim=0), -1)
        cap_emb = torch.bmm(span_marg.unsqueeze(-2),  cap_feats).squeeze(-2)
        cap_emb = utils.l2norm(cap_emb)
        # split transitive and intransitive caps and images
        img_emb_tr, img_emb_intr = torch.split(img_emb, bsize)
        cap_emb_tr, cap_emb_intr = torch.split(cap_emb, bsize)
        # compare cosine similarity with images
        itr_ctr = sim(img_emb_tr, cap_emb_tr)
        itr_cintr = sim(img_emb_tr, cap_emb_intr)
        iintr_cintr = sim(img_emb_intr, cap_emb_intr)
        iintr_ctr = sim(img_emb_intr, cap_emb_tr)
        ans_tr = torch.gt(itr_ctr,itr_cintr)
        ans_intr = torch.gt(iintr_cintr,iintr_ctr)
        ids_transitives += ids_tr
        ids_intransitives += ids_intr
        itr_ctr_all += itr_ctr.tolist()
        itr_cintr_all += itr_cintr.tolist()
        iintr_cintr_all += iintr_cintr.tolist()
        iintr_ctr_all += iintr_ctr.tolist()
        ans_transitives += ans_tr.tolist()
        ans_intransitives += ans_intr.tolist()
        del images, captions, lengths, spans
    n = len(ans_transitives)
    tr_score = sum(ans_transitives) / n
    intr_score = sum(ans_intransitives) / n
    score = (tr_score + intr_score) / 2
    info = '\nImage match score: {:.4f}, Transitive score: {:.4f}, Intransitive score: {:.4f} '
    info = info.format(score, tr_score, intr_score)
    logger.info(info)
    if save:
        file = opt.logger_name + '/syntactic_bootstrapping_results/' + str(current_epoch) +'.csv'
        utils.save_columns_to_csv(file, ids_transitives, ids_intransitives, itr_ctr_all, itr_cintr_all, iintr_cintr_all, iintr_ctr_all, ans_transitives, ans_intransitives)
    return score