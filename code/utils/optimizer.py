from torch import optim

def get_optimizer(model, cfg):
    params = model.parameters()
    optimizer = optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, eps=1e-5)

    return optimizer