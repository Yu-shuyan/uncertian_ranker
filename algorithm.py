import math

import numpy as np
import torch
from torch import nn
from model import *
import os
os.environ['CURL_CA_BUNDLE'] = ''


def init_algorithm(args, preselect=False):

    # if 'MATH' in args.data_root_test[0]:
    #     num_train_domains = 7
    #
    # else:
    num_train_domains = 1

    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")  # one GPU



    # Main model
    if preselect:
        model = policy_network(model_config=args.model_config, embedding_size=args.embedding_size,
                               freeze_encoder=True)
    else:
        ## policy network
        model = policy_network(model_config=args.model_config,
                                   add_linear=True,
                                   embedding_size=args.embedding_size,
                                   freeze_encoder=True)

    model = model.to(device)


    # Loss fn

    loss_fn = nn.CrossEntropyLoss()

    # Algorithm
    hparams = {}
    hparams['support_size'] = args.batch_size // args.meta_batch_size
    if args.algorithm == 'ERM' or args.algorithm == 'frozen':
        hparams['batch_size'] = args.batch_size
        algorithm = ERM(model, loss_fn, device, 'adam', args.lr, hparams=hparams)
    else:
        raise Exception("The algorithm does not exist!")
    return algorithm




class ERM(nn.Module):
    def __init__(self, model, loss_fn, device, optimizer_name, lr, init_optim=True,hparams={}, **kwargs):
        super().__init__()
        self.batch_size = hparams["batch_size"]
        self.model = model
        self.loss_fn = loss_fn
        self.device = device

        self.optimizer_name = optimizer_name
        self.learning_rate = lr


        if init_optim:
            params = self.model.parameters()
            self.init_optimizers(params)

    def init_optimizers(self, params):
        if self.optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        else:
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, params),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay)

    def predict(self, x, test=False):
        n_batch = math.ceil(len(x) / self.batch_size)
        logits = []
        for batch_id in range(n_batch):
            start = batch_id * self.batch_size
            end = start + self.batch_size
            end = min(len(x), end)  # in case final domain has fewer than support size samples

            domain_x = x[start:end]
            domain_logits = self.model(domain_x)
            logits.append(domain_logits)
        logits = torch.cat(logits)
        return logits

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_acc(self, logits, labels):
        # Evaluate
        preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
        accuracy = np.mean(preds == labels.detach().cpu().numpy().reshape(-1))
        return accuracy

    def learn(self, images, labels, group_ids=None):

        self.train()

        # Forward
        logits = self.predict(images)
        loss = self.loss_fn(logits, labels)


        self.update(loss)

        stats = {
                 'objective': loss.detach().item(),
                }

        return logits, stats















