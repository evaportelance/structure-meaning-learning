import os
import time, pickle, argparse, logging
import numpy as np
import torch
#from transformers import get_scheduler, AutoTokenizer, AutoModelForCausalLM, AutoConfig, OPTConfig

import vpcfg.as_dataloader as data
from vpcfg.utils import Vocabulary, save_checkpoint
from vpcfg.as_evaluation import AverageMeter, LogCollector, semantic_bootstrapping_test, syntactic_bootstrapping_test
from vpcfg.as_vocab import get_vocab

def train(opt, train_loader, model, epoch, val_loader):
    # average meters to record the training statistics
    train_logger = LogCollector()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    nbatch = len(train_loader)
    # switch to train mode
    end = time.time()
    model.n_word = 0
    model.n_sent = 0
    model.s_time = end
    model.all_stats = [[0., 0., 0.]]
    for i, train_data in enumerate(train_loader):
        # Always reset to train mode
        model.train()
        # measure data loading time
        data_time.update(time.time() - end)
        # make sure train logger is used
        model.logger = train_logger
        # Update the model
        info = model.forward(*train_data, epoch=epoch)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # Print log info
        if model.niter % opt.log_step == 0:
            logger.info(
                'Epoch: [{0}][{1}/{2}] {e_log} {info}'
                .format(
                    epoch, i, nbatch, e_log=str(model.logger), info=info
                )
            )
        #break
        # validate at every val_step
        if model.niter % opt.val_step == 0:
            semantic_bootstrapping_test(opt, val_loader, model, logger, epoch, save=False)

if __name__ == '__main__':
    # hyper parameters
    parser = argparse.ArgumentParser()

    # Parser: Generative model parameters
    
    parser.add_argument('--z_dim', default=64, type=int, help='latent dimension')
    parser.add_argument('--t_states', default=60, type=int, help='number of preterminal states')
    parser.add_argument('--nt_states', default=30, type=int, help='number of nonterminal states')
    parser.add_argument('--state_dim', default=256, type=int, help='symbol embedding dimension')
    # Parser: Inference network parameters
    parser.add_argument('--h_dim', default=768, type=int, help='hidden dim for variational LSTM')
    parser.add_argument('--w_dim', default=768, type=int, help='embedding dim for variational LSTM')
    parser.add_argument('--gpu', default=1, type=int, help='which gpu to use')

    #
    parser.add_argument('--seed', default=527, type=int, help='random seed')
    parser.add_argument('--model_init', default=None, type=str, help='checkpoint to initialize model with')
    #parser.add_argument('--w2vec_file', default=None, type=str, help='word vector file')
    parser.add_argument('--max_length', default=40, type=int, help='max sentence length')
    parser.add_argument('--prefix', default="all", type=str, help='prefix')
    #parser.add_argument('--parser_type', default='2nd', type=str, help='model name (1st/2nd)')
    #parser.add_argument('--share_w2vec', default=False, type=bool, help='shared embeddings')
    parser.add_argument('--visual_mode', action='store_true', help='run visual model')
    
    parser.add_argument('--encoder_file', default=None, help='image representations file name to use')
    parser.add_argument('--tiny', action='store_true', help='if testing will create tiny dataloaders')
    parser.add_argument('--one_shot', action='store_true', help='If split for ssyntactic bootstrapping test should be one-shot')
    parser.add_argument('--shuffle', action='store_true', help='shuffle training data')

    #
    parser.add_argument('--sem_dim', default=768, type=int, help='semantic rep. dim')
    parser.add_argument('--syn_dim', default=768, type=int, help='syntactic rep. dim')
    parser.add_argument('--word_dim', default=768, type=int,
                        help='dimensionality of the word embedding')
    parser.add_argument('--lstm_dim', default=768, type=int,
                        help='dimensionality of the lstm hidden embedding')
    
    parser.add_argument('--vocab_size', default=2000, type=int,
                        help='tokenizer/vocabulary size')
    parser.add_argument('--data_path', default='../preprocessed-data/abstractscenes', help='path to datasets')
    #parser.add_argument('--tokenizer_path', default='../../babylm-models/test/', help='path to pretrained tokenizer')
    #parser.add_argument('--save_model_path', default='../../babylm-models/test/', help='path to directory with model configs')
    parser.add_argument('--logger_name', default='../../../scratch/vcpcfg/runs/test', help='location for model outputs and logfiles to be saved')
    
    parser.add_argument('--margin', default=0.2, type=float,
                        help='rank loss margin')
    parser.add_argument('--num_epochs', default=100, type=int,
                        help='number of training epochs')
    parser.add_argument('--batch_size', default=5, type=int,
                        help='size of a training mini-batch')
    parser.add_argument('--grad_clip', default=3., type=float,
                        help='gradient clipping threshold')
    parser.add_argument('--lr', default=.001, type=float,
                        help='initial learning rate')
    parser.add_argument('--workers', default=1, type=int,
                        help='number of data loader workers')
    #
    parser.add_argument('--log_step', default=500, type=int,
                        help='number of steps to print and record the log')
    parser.add_argument('--val_step', default=float("inf"), type=int,
                        help='number of steps to run validation')
    #
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='dimensionality of the image embedding')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    #
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='optimizer, can be Adam, SGD, etc.')
    parser.add_argument('--beta1', default=0.75, type=float, help='beta1 for adam')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta2 for adam')
    #
    parser.add_argument('--vse_mt_alpha', type=float, default=1.0, help='weight parameter for matching loss')
    parser.add_argument('--vse_lm_alpha', type=float, default=1.0,  help='weight parameter for  loss')
    parser.add_argument('--sem_first', action='store_true', help='Run semantics first model')
    parser.add_argument('--syn_first', action='store_true', help='Run syntax first model')

    opt = parser.parse_args()
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    # setup logger
    if os.path.exists(opt.logger_name):
        print(f'Warning: the folder {opt.logger_name} exists.')
    else:
        print('Creating {}'.format(opt.logger_name))
        os.mkdir(opt.logger_name)
        os.mkdir(opt.logger_name+'/syntactic_bootstrapping_results')
        os.mkdir(opt.logger_name+'/semantic_bootstrapping_results')
        os.mkdir(opt.logger_name+'/checkpoints')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(os.path.join(opt.logger_name, 'train.log'), 'w')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.propagate = False
    logger.info('cuda:{}@{}'.format(opt.gpu, os.uname().nodename))
    logger.info(opt)

    # load predefined vocabulary and pretrained word embeddings if applicable
    #vocab = pickle.load(open(os.path.join(opt.data_path, opt.vocab_name), 'rb'))
    ## EP to get Abstract Scenes vocabulary
    vocab = get_vocab(opt.data_path)
    opt.vocab_size = len(vocab)
    logger.info("|vocab|={}".format(len(vocab)))

    # construct the model
    if not opt.visual_mode:
        from vpcfg.model import VGCPCFGs 
    else:
        logger.info("using visually-grounded version of model.")
        from vpcfg.model_vis import VGCPCFGs 
    sampler = True
    model = VGCPCFGs(opt, vocab, logger)
    if opt.model_init:
        logger.info("override parser's params.")
        checkpoint = torch.load(opt.model_init, map_location='cpu')
        parser_params = checkpoint['model'][Model.NS_PARSER]
        model.parser.load_state_dict(parser_params)
        start_epoch = checkpoint['epoch'] + 1
    else:
        save_checkpoint({
        'epoch': -1,
        'model': model.get_state_dict(),
        'best_rsum': -1,
        'opt': opt,
        'Eiters': -1 }, False, -1, prefix=opt.logger_name)
        start_epoch = 0

    # Load data loaders
    data.set_constant(opt.visual_mode, opt.max_length)
    train_loader, syn_test_loader, sem_test_loader = data.get_data_iters(opt.data_path, opt.prefix, vocab, opt.batch_size, opt.workers, load_img=opt.visual_mode, encoder_file=opt.encoder_file, img_dim=opt.img_dim, shuffle=opt.shuffle, sampler=sampler, tiny=opt.tiny, one_shot=opt.one_shot)
    logger.info("Number of train items: {}, semantic test items: {}, syntactic test items: {}".format(train_loader.dataset.length, sem_test_loader.dataset.length, syn_test_loader.dataset.length))
    
    if start_epoch == 0:
        semantic_bootstrapping_test(opt, sem_test_loader, model, logger, -1)
        syntactic_bootstrapping_test(opt, syn_test_loader, model, logger, -1)
    best_rsum = float('inf')
    
    if opt.sem_first:
        model.vse_mt_alpha = 1.0
        model.vse_lm_alpha = 0.0
        logger.info("Training model on semantics loss first.")
        for epoch in range(int(opt.num_epochs/2)):
            current_epoch = start_epoch + epoch
            # train for one epoch
            train(opt, train_loader, model, epoch, syn_test_loader)
            # evaluate on validation set using VSE metrics
            rsum = semantic_bootstrapping_test(opt, sem_test_loader, model, logger, current_epoch)
            score = syntactic_bootstrapping_test(opt, syn_test_loader, model, logger, current_epoch)
            # remember best R@ sum and save checkpoint
            is_best = rsum < best_rsum
            best_rsum = max(rsum, best_rsum)
            save_checkpoint({
                'epoch': current_epoch,
                'model': model.get_state_dict(),
                'best_rsum': best_rsum,
                'opt': opt,
                'Eiters': model.niter,
            }, is_best, current_epoch, prefix=opt.logger_name)
        model.vse_mt_alpha = 1.0
        model.vse_lm_alpha = 1.0
        logger.info("Training model on syntax loss second.")
        for epoch in range(int(opt.num_epochs/2), opt.num_epochs):
            current_epoch = start_epoch + epoch
            # train for one epoch
            train(opt, train_loader, model, epoch, syn_test_loader)
            # evaluate on validation set using VSE metrics
            rsum = semantic_bootstrapping_test(opt, sem_test_loader, model, logger, current_epoch)
            score = syntactic_bootstrapping_test(opt, syn_test_loader, model, logger, current_epoch)
            # remember best R@ sum and save checkpoint
            is_best = rsum < best_rsum
            best_rsum = max(rsum, best_rsum)
            save_checkpoint({
                'epoch': current_epoch,
                'model': model.get_state_dict(),
                'best_rsum': best_rsum,
                'opt': opt,
                'Eiters': model.niter,
            }, is_best, current_epoch, prefix=opt.logger_name)
        
    elif opt.syn_first:
        model.vse_mt_alpha = 0.0
        model.vse_lm_alpha = 1.0
        logger.info("Training model on syntax loss first.")
        for epoch in range(int(opt.num_epochs/2)):
            current_epoch = start_epoch + epoch
            # train for one epoch
            train(opt, train_loader, model, epoch, syn_test_loader)
            # evaluate on validation set using VSE metrics
            rsum = semantic_bootstrapping_test(opt, sem_test_loader, model, logger, current_epoch)
            score = syntactic_bootstrapping_test(opt, syn_test_loader, model, logger, current_epoch)
            # remember best R@ sum and save checkpoint
            is_best = rsum < best_rsum
            best_rsum = max(rsum, best_rsum)
            save_checkpoint({
                'epoch': current_epoch,
                'model': model.get_state_dict(),
                'best_rsum': best_rsum,
                'opt': opt,
                'Eiters': model.niter,
            }, is_best, current_epoch, prefix=opt.logger_name)
        model.vse_mt_alpha = 1.0
        model.vse_lm_alpha = 1.0
        logger.info("Training model on semantics loss second.")
        for epoch in range(int(opt.num_epochs/2), opt.num_epochs):
            current_epoch = start_epoch + epoch
            # train for one epoch
            train(opt, train_loader, model, epoch, syn_test_loader)
            # evaluate on validation set using VSE metrics
            rsum = semantic_bootstrapping_test(opt, sem_test_loader, model, logger, current_epoch)
            score = syntactic_bootstrapping_test(opt, syn_test_loader, model, logger, current_epoch)
            # remember best R@ sum and save checkpoint
            is_best = rsum < best_rsum
            best_rsum = max(rsum, best_rsum)
            save_checkpoint({
                'epoch': current_epoch,
                'model': model.get_state_dict(),
                'best_rsum': best_rsum,
                'opt': opt,
                'Eiters': model.niter,
            }, is_best, current_epoch, prefix=opt.logger_name)
    else:
        for epoch in range(opt.num_epochs):
            current_epoch = start_epoch + epoch
            # train for one epoch
            train(opt, train_loader, model, epoch, syn_test_loader)
            # evaluate on validation set using VSE metrics
            rsum = semantic_bootstrapping_test(opt, sem_test_loader, model, logger, current_epoch)
            score = syntactic_bootstrapping_test(opt, syn_test_loader, model, logger, current_epoch)
            # remember best R@ sum and save checkpoint
            is_best = rsum < best_rsum
            best_rsum = max(rsum, best_rsum)
            save_checkpoint({
                'epoch': current_epoch,
                'model': model.get_state_dict(),
                'best_rsum': best_rsum,
                'opt': opt,
                'Eiters': model.niter,
            }, is_best, current_epoch, prefix=opt.logger_name)
