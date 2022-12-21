import torch
import torch.nn as nn
import numpy as np
import random

def set_seed(seed) :
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


class CPL_Labeler :
    def __init__(self, config, num_classes, wandb) -> None:
        self.max_threshold = config.max_threshold
        self.num_classes = num_classes
        self.use_warmup = config.use_warmup
        self.nonlinear_mapping = config.nonlinear_mapping
        self.wandb = wandb
        
    def get_threshold(self, weak_logits) :
        """
        weak_logits : [batch_size, num_classes]
        return : [num_classes]
        """
        bsz = weak_logits.shape[0]
        over_max_threshold = weak_logits > self.max_threshold
        num_over_max_threshold = over_max_threshold.sum(dim = 0)
        max_num = num_over_max_threshold.max()
        if self.use_warmup :
            deno = max_num if max_num > bsz - max_num else bsz - max_num
        else : 
            deno = max_num
        beta = num_over_max_threshold / deno

        flex_threshold = self.max_threshold * beta

        if self.nonlinear_mapping :
            flex_threshold = flex_threshold/(2-flex_threshold)

        return flex_threshold

    def __call__(self, weak_logits, weak_preds) :
        """
        weak_logits : [batch_size, num_classes]
        return : [batch_size, num_classes]
        """
        threshold = self.get_threshold(weak_logits)
        pseudo_labels = torch.tensor([pred if logits.max() > threshold[pred] else -100 for logits, pred in zip(weak_logits, weak_preds)]) # -100 is a ignore index for Cross Entorpy Loss; not calculated in loss
        return pseudo_labels


def main() :
    class Config :
        max_threshold = 0.9
        num_classes = 3
        use_warmup = False
        nonlinear_mapping = True

    config = Config()
    cpl_labeler = CPL_Labeler(config)

    logits = torch.rand(100, 3) # [batch_size, num_classes]
    logits = torch.nn.Softmax(dim = 1)(logits)
    preds = logits.argmax(dim = 1)

    pseudo_labels, threshold = cpl_labeler(logits, preds)
    print(pseudo_labels)
    print(threshold)

if __name__ == "__main__" :
    main()