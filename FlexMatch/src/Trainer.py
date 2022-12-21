import wandb

from transformers import Trainer
from transformers import AutoModelForImageClassification
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW
from tqdm import tqdm, trange
import torch

from utils import CPL_Labeler

class FlexMatchTrainer(Trainer) :
    def __init__(
        self,
        trainer_config,
        cpl_labeler_config, 
        train_dataset : Dataset,
        unlabeled_dataset : Dataset, 
        test_dataset : Dataset,
        wandb = wandb
        ) -> None:

        self.device = trainer_config.device

        self.batch_size = trainer_config.batch_size
        self.k = trainer_config.k # number of unlabeled samples per labeled sample
        self.test_batch_size = trainer_config.test_batch_size
        self.total_epoch = trainer_config.total_epoch
        self.lam1 = trainer_config.lam1

        self.model = AutoModelForImageClassification.from_pretrained(trainer_config.model_name, num_labels = trainer_config.num_classes, ignore_mismatched_sizes = True).to(self.device)
        self.train_dataset = train_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.test_dataset = test_dataset

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = 4,
            drop_last = True,
        )

        self.unlabeled_dataloader = DataLoader(
            self.unlabeled_dataset,
            batch_size = self.batch_size * self.k,
            shuffle = True,
            num_workers = 4,
            drop_last = True,
        )

        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size = self.test_batch_size,
            shuffle = False,
            num_workers = 4,
            drop_last = True,
        )

        self.optimizer = AdamW(
            self.model.parameters(),
            lr = trainer_config.learning_rate,
        )

        self.cpl_labeler = CPL_Labeler(cpl_labeler_config, trainer_config.num_classes, wandb)
        self.wandb = wandb

    def train(self) :
        self.model.train()
        train_iterator = trange(
            self.total_epoch,
            desc = "Epoch",
            disable = False,
        )
        total_iter = 0
        for epoch in train_iterator :
            for (train_idx, train_X, train_y), (ulb_idx, ulb_weak, ulb_strong, ulb_y) in zip(self.train_dataloader, self.unlabeled_dataloader) :
                total_iter += 1 
                train_X, train_y = train_X.to(self.device), train_y.to(self.device)
                ulb_weak, ulb_strong = ulb_weak.to(self.device), ulb_strong.to(self.device)
                self.optimizer.zero_grad()
                loss = self.training_step(self.model, train_X, train_y, ulb_weak, ulb_strong, ulb_y, total_iter)
                loss.backward()
                self.optimizer.step()

    def training_step(self, model: nn.Module, train_inputs, train_y, ulb_weak, ulb_strong, ulb_y, total_iter) -> torch.Tensor:
        model_inputs = torch.concat((train_inputs, ulb_weak, ulb_strong), dim = 0)
        model_outputs = model(model_inputs)["logits"] # Sofmax 통과 후의 output output : ["prediction" : [batch_size,], "logits" : [batch_size, num_classes]]
        train_output, ulb_weak_output, ulb_strong_output = model_outputs[:self.batch_size], model_outputs[self.batch_size:self.batch_size*(self.k+1)], model_outputs[self.batch_size*(self.k+1):]
        ulb_strong_pred = torch.argmax(ulb_strong_output, dim = 1)
        train_loss = self.loss_fn(train_output, train_y)
        pseudo_label = self.cpl_labeler(ulb_weak_output, ulb_strong_pred).to(self.device)
        ulb_loss = self.loss_fn(ulb_strong_output, pseudo_label)
        loss = train_loss + self.lam1*ulb_loss
        ulb_pseudo_acc = self.acc_fn(ulb_strong_pred, pseudo_label)
        ulb_acc = self.acc_fn(ulb_strong_pred, ulb_y)
        train_acc = self.acc_fn(train_output.argmax(dim = 1), train_y)
        self.wandb.log({
                "total_loss" : loss.item(),
                "lab_loss" : train_loss.item(),
                "ulb_loss" : ulb_loss.item(),
                "num_over_threshold" : sum(pseudo_label != -100).item()/pseudo_label.size(0),
                "ulb_acc" : ulb_acc.item(),
                "ulb_pseudo_acc" : ulb_pseudo_acc.item(),
                "lab_acc" : train_acc.item()
                }, step = total_iter)
        return loss

    def loss_fn(self, logits, labels) :
        return nn.CrossEntropyLoss()(logits, labels)

    def acc_fn(self, prediction, labels) :
        prediction = prediction.detach().cpu()
        labels = labels.detach().cpu()
        return (prediction == labels).float().mean()

    def test(self) :
        self.model.eval()
        test_iterator = trange(
            len(self.test_dataloader),
            desc = "Test",
            disable = False,
        )
        test_acc_total = 0

        with torch.no_grad() :
            for test_idx, test_X, test_y in self.test_dataloader :
                test_X, test_y = test_X.to(self.device), test_y.to(self.device)
                test_output = self.model(test_X)
                test_loss = self.loss_fn(test_output["logits"], test_y)
                test_pred = test_output["logits"].argmax(dim = 1)
                test_acc = self.acc_fn(test_pred, test_y)
                test_iterator.set_postfix(
                    loss = test_loss.item(),
                    acc = test_acc.item(),
                )
                test_iterator.update()
                test_acc_total += test_acc.item()
        test_acc_total /= len(self.test_dataloader)
        self.wandb.log({"test_acc" : test_acc_total})
