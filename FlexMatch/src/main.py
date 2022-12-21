import sys

from transformers import HfArgumentParser
from dataclasses import dataclass
import wandb

from Trainer import FlexMatchTrainer
from dataset import FlexMatchDataset
from utils import set_seed

@dataclass
class DatasetConfig :
    data_dir : str = "./dataset/SSL_data"
    num_labeled_data : int = 1000
    num_unlabeled_data : int = 4000

@dataclass
class TrainerConfig :
    batch_size : int = 64
    k : int = 4
    test_batch_size : int = 128
    total_epoch : int = 1
    learning_rate : float = 1e-5
    model_name : str = "microsoft/resnet-18"
    num_classes : int = 10
    device : str = "cuda:0"
    lam1 : float = 1.0

@dataclass
class CPLLabelerConfig :
    use_warmup : bool = True
    nonlinear_mapping : bool = True
    max_threshold : float = 0.95

@dataclass
class WandbConfig :
    project_name : str = "FlexMatch"
    run_name : str = "debug"


def main() :
    set_seed(42)
    parser = HfArgumentParser((DatasetConfig, TrainerConfig, CPLLabelerConfig, WandbConfig))
    if len(sys.argv) == 2 and sys.argv[1].endswith("yaml") :
        print("\n", f">>> Loading config file: {sys.argv[1]}")
        dataset_config, trainer_config, cpl_labeler_config, wandb_config = parser.parse_yaml_file(sys.argv[1])
    else :
        print(f">>> Loading default config file: ./debug.yaml")
        dataset_config, trainer_config, cpl_labeler_config, wandb_config = parser.parse_yaml_file("./debug.yaml")

    print(f">>> dataset_config: {dataset_config}")
    print(f">>> trainer_config: {trainer_config}")
    print(f">>> cpl_labeler_config: {cpl_labeler_config}")
    print(f">>> wandb_config: {wandb_config}")

    train_dataset = FlexMatchDataset(dataset_config)
    ulb_dataset = FlexMatchDataset(dataset_config, mode="unlabeled")
    test_dataset = FlexMatchDataset(dataset_config, mode="test")

    wandb.init(project = wandb_config.project_name, name = wandb_config.run_name)
    # wandb.config(sys.argv[1])

    trainer = FlexMatchTrainer(
        trainer_config,
        cpl_labeler_config,
        train_dataset,
        ulb_dataset,
        test_dataset,
        wandb = wandb
    )

    trainer.train()
    trainer.test()
if __name__ == "__main__" :
    main()