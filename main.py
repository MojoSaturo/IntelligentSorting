import argparse
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="datasets/trashnet", type=str)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--val_split", default=0.2, type=float)
parser.add_argument("--epochs", default=150, type=int)
parser.add_argument("--learning_rate", default=1e-3, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--weight_decay", default=1e-3, type=float)

parser.add_argument("--saving_interval", default=None, type=int)
parser.add_argument("--saving_path", default=None, type=str)
parser.add_argument("--scheduler_milestones", nargs="*", default=None, type=int)
parser.add_argument("--scheduler_gamma", default=0.1, type=float)
parser.add_argument("--logging_path", default=None, type=str)
parser.add_argument("--state_path", default=None, type=str)
args = parser.parse_args()


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size):
    setup(rank=rank, world_size=world_size)
    torch.manual_seed(4444)
    np.random.seed(4444)
    train_loader, val_loader = load_data()
    trainer = Trainer(
        rank=rank,
        world_size=world_size,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    trainer.train(args.epochs)
    cleanup()


class Trainer:
    def __init__(
        self,
        rank,
        world_size,
        train_loader,
        val_loader,
    ):
        self.rank = rank
        self.world_size = world_size
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = models.swin_v2_t(weights="IMAGENET1K_V1")
        self.model.head = nn.Linear(768, 6)
        self.model.to(self.rank)
        self.ddp_model = DDP(self.model, device_ids=[self.rank])
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            params=self.ddp_model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        if args.scheduler_milestones is not None:
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=self.optimizer,
                milestones=args.scheduler_milestones,
                gamma=args.scheduler_gamma,
            )
        self.state_path = args.state_path
        self.saving_path = args.saving_path
        self.logging_path = args.logging_path
        if args.state_path is not None:
            state = torch.load(args.state_path, map_location=torch.device(self.rank))
            self.model.load_state_dict(state["model_state_dict"])
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            if args.scheduler_milestones is not None:
                self.scheduler.load_state_dict(state["scheduler_state_dict"])
        if args.logging_path is not None:
            if not os.path.exists(args.logging_path):
                os.makedirs(args.logging_path)
            self.writer = SummaryWriter(args.logging_path)

    def _train_epoch(self, epoch):
        running_loss, running_acc = 0, 0
        for step, (images, labels) in enumerate(self.train_loader):
            if self.rank == 0:
                print(
                    f"Epoch: {epoch}  [{step + 1}/{len(self.train_loader)}]\r",
                    end="",
                )
            self.optimizer.zero_grad()
            outputs = self.ddp_model(images.to(self.rank))
            loss = self.criterion(outputs, labels.to(self.rank))
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            running_acc += (outputs.argmax(1) == labels.to(self.rank)).sum().item()
        loss_avg = running_loss / len(self.train_loader)
        acc_avg = running_acc * 100 / len(self.train_loader.dataset)
        return loss_avg * self.world_size, acc_avg * self.world_size

    def _val_epoch(self):
        with torch.no_grad():
            running_loss, running_acc = 0, 0
            for images, labels in self.val_loader:
                outputs = self.ddp_model(images.to(self.rank))
                loss = self.criterion(outputs, labels.to(self.rank))
                running_loss += loss.item()
                running_acc += (outputs.argmax(1) == labels.to(self.rank)).sum().item()
            loss_avg = running_loss / len(self.val_loader)
            acc_avg = running_acc * 100 / len(self.val_loader.dataset)
            return loss_avg * self.world_size, acc_avg * self.world_size

    def _save_checkpoint(self, epoch):
        if args.saving_interval is not None and args.saving_path is not None:
            if not os.path.exists(args.saving_path):
                os.makedirs(args.saving_path)
            state_dict = {
                "model_state_dict": self.ddp_model.module.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }
            if args.scheduler_milestones is not None:
                state_dict.update({"scheduler_state_dict": self.scheduler.state_dict()})
            torch.save(
                state_dict,
                args.saving_path + f"/checkpoint_{epoch}.pth",
            )
            print("Checkpoint Saved!")

    def train(self, epochs):
        time_total = 0
        self.ddp_model.train()
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            train_loss, train_acc = self._train_epoch(epoch)
            if args.scheduler_milestones is not None:
                self.scheduler.step()
            if self.rank == 0:
                if self.val_loader is not None:
                    val_loss, val_acc = self._val_epoch()
                else:
                    val_loss, val_acc = 0, 0
                loss_dict = {"Train": train_loss}
                acc_dict = {"Train": train_acc}
                if self.val_loader is not None:
                    loss_dict.update({"Val": val_loss})
                    acc_dict.update({"Val": val_acc})
                if args.logging_path is not None:
                    self.writer.add_scalars("Loss", loss_dict, epoch - 1)
                    self.writer.add_scalars("Accuracy", acc_dict, epoch - 1)
                if args.saving_interval is not None and args.saving_path is not None:
                    if epoch % args.saving_interval == 0:
                        self._save_checkpoint(epoch)
                time_cost = time.time() - start_time
                time_total += time_cost
                time_avg = time_total / epoch
                time_1, time_2 = time_avg * epoch / 3600, time_avg * epochs / 3600
                print(
                    f"Epoch: {epoch}  "
                    + f"[{self.optimizer.param_groups[0]['lr']:.0e}]  "
                    + f"[{train_loss:.4f}|{val_loss:.4f}]  "
                    + f"[{train_acc:.2f}%|{val_acc:.2f}%]  "
                    + f"[{time_cost:.2f}s]  "
                    + f"[{time_1:.2f}/{time_2:.2f}h]"
                    + f"[{time_2 - time_1:.2f}h]"
                )
        if args.logging_path is not None:
            self.writer.close()


def load_data():
    transform = transforms.Compose(
        [
            transforms.Resize(260, Image.Resampling.BICUBIC),
            transforms.CenterCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    data = ImageFolder(root=args.data_path, transform=transform)
    val_data_split = int((len(data) * args.val_split))
    train_data_split = len(data) - val_data_split
    train_data, val_data = random_split(
        dataset=data, lengths=[train_data_split, val_data_split]
    )
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=DistributedSampler(train_data),
        pin_memory=True,
    )
    if args.val_split != 0:
        val_loader = DataLoader(
            dataset=val_data,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=DistributedSampler(val_data),
            pin_memory=True,
        )
    else:
        val_loader = None
    return train_loader, val_loader


if __name__ == "__main__":
    torch.cuda.empty_cache()
    world_size = torch.cuda.device_count()
    mp.spawn(
        fn=main,
        args=(world_size,),
        nprocs=world_size,
    )
