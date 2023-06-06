import torch
from collections import defaultdict
import random

class ASAM:
    def __init__(self, optimizer, model, rho=0.5, eta=0.01, cfg=None, momentum_grad=None):
        self.optimizer = optimizer
        self.model = model
        self.cfg = cfg
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)
        self.momentum_grad = momentum_grad

    @torch.no_grad()
    def ascent_step(self):
        wgrads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if t_w is None:
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if 'weight' in n:
                t_w[...] = p[...]
                t_w.abs_().add_(self.eta)
                p.grad.mul_(t_w)
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-8
        
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if 'weight' in n:
                p.grad.mul_(t_w)
            eps = t_w
            eps[...] = p.grad[...]
            eps.mul_(self.rho / wgrad_norm)
            if self.cfg.sparse is True:
                eps.data = eps.data * self.state[n]['local_mask'].to(p.data.device)
            self.state[p]["eps"] = eps
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()

    @torch.no_grad()
    def initiate_mask_relevant(self):
        all_scores = []
        momentum_grad = {}
        for n, p in self.model.named_parameters():
            score = torch.abs(p.grad.clone()).data
            all_scores.append(score)
            momentum_grad[n] = p.grad.clone().data
        all_scores = torch.cat([torch.flatten(x) for x in all_scores])

        topk_num = int(len(all_scores) * self.cfg.topkgrad)
        g_value, g_index = torch.topk(all_scores, topk_num)
        topk_mask_list = torch.zeros_like(all_scores)
        topk_mask_list.scatter_(0, g_index, torch.ones_like(g_value))
        topk_mask_list = torch.tensor(topk_mask_list, dtype=torch.uint8)

        start_index = 0
        for n, p in self.model.named_parameters():
            growth_mask = topk_mask_list[start_index: start_index + p.numel()].reshape(p.shape)
            self.state[n]['local_mask'] = growth_mask.to(p.data.device)
            self.state[n]['local_mask'].require_grad = False
            start_index = start_index + p.numel()

        return momentum_grad

    @torch.no_grad()
    def update_mask_and_momentum_grad(self, global_mask, momentum_grad):
        all_scores = []
        for n, p in self.model.named_parameters():
            score = torch.abs(p.grad.clone()).data
            momentum_grad[n] = self.cfg.momentum_gamma * momentum_grad[n] + (1 - self.cfg.momentum_gamma) * score
            all_scores.append(momentum_grad[n])
        all_scores = torch.cat([torch.flatten(x) for x in all_scores])

        topk_num = int(len(all_scores) * self.cfg.topkgrad)
        g_value, g_index = torch.topk(all_scores, topk_num)
        mask_list_tmp = torch.zeros_like(all_scores)
        mask_list_tmp.scatter_(0, g_index, torch.ones_like(g_value))
        mask_list_tmp = torch.tensor(mask_list_tmp, dtype=torch.uint8)

        start_index = 0
        local_mask_tmp = {}
        for n, p in self.model.named_parameters():
            local_mask_tmp[n] = mask_list_tmp[start_index: start_index + p.numel()].reshape(p.shape)
            start_index = start_index + p.numel()
        
        nonintersec_index = []
        start_index_tmp = 0
        for k,v in local_mask_tmp.items():
            local_mask_tmp[k] = v.to('cpu')

        for n, p in self.model.named_parameters():
            local_vec = torch.flatten(local_mask_tmp[n])
            global_vec = torch.flatten(global_mask[n.replace('module.', '')])
            global_local_mask_index = torch.where( (local_vec==0) & (global_vec==0) )
            random_index_of_index = torch.tensor( random.sample( list(range(len(global_local_mask_index[0]))), min( int(self.cfg.nonintersec_p * len(local_vec)), int(len(global_local_mask_index[0]) ) ) ) , dtype=torch.int64)
            global_local_mask_index_final = torch.index_select(global_local_mask_index[0], 0, random_index_of_index)
            nonintersec_index.append(start_index_tmp + global_local_mask_index_final)
            start_index_tmp = start_index_tmp + p.numel()

        g_index = torch.cat( (g_index, torch.cat(nonintersec_index).to(self.cfg.device)) )
        g_index = torch.tensor(g_index, dtype=torch.int64)
        mask_list = torch.zeros_like(all_scores, dtype=torch.int64).to(self.cfg.device)
        mask_list.scatter_(0, g_index, torch.ones_like(g_index, dtype=torch.int64))

        start_index = 0
        for n, p in self.model.named_parameters():
            growth_mask = mask_list[start_index: start_index + p.numel()].reshape(p.shape)
            self.state[n]['local_mask'] = growth_mask.to(p.data.device)
            self.state[n]['local_mask'].require_grad = False
            start_index = start_index + p.numel()

        return momentum_grad

    @torch.no_grad()
    def get_local_mask(self, momentum_grad):

        all_scores = []
        for n, p in self.model.named_parameters():
            all_scores.append(momentum_grad[n])
        all_scores = torch.cat([torch.flatten(x) for x in all_scores])

        topk_num = int(len(all_scores) * self.cfg.topkgrad)
        g_value, g_index = torch.topk(all_scores, topk_num)
        topk_mask_list = torch.zeros_like(all_scores)
        topk_mask_list.scatter_(0, g_index, torch.ones_like(g_value))
        topk_mask_list = torch.tensor(topk_mask_list, dtype=torch.uint8)

        start_index = 0
        local_mask = {}
        for n, p in self.model.named_parameters():
            growth_mask = topk_mask_list[start_index: start_index + p.numel()].reshape(p.shape)
            local_mask[n] = growth_mask
            local_mask[n] = local_mask[n].to('cpu')
            local_mask[n].require_grad = False
            start_index = start_index + p.numel()

        return local_mask

    @torch.no_grad()
    def get_history_mask(self, local_mask):

        for n, p in self.model.named_parameters():
            self.state[n]['local_mask'] = local_mask[n]
            self.state[n]['local_mask'].require_grad = False