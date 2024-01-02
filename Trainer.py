import torch as T
from tqdm.auto import tqdm
from typing import Callable


class TrainerV1:
    model:     T.nn.Module
    optimizer: T.optim.Optimizer
    scheduler: T.optim.lr_scheduler
    criterion: Callable

    def __init__(self, model: T.nn.Module,
                 criterion: Callable,
                 optimizer: T.optim.Optimizer,
                 scheduler: T.optim.lr_scheduler):
        self.model     = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

    def forward(self, input_loader, progress_bar=False) -> float:
        loss_total = 0
        device     = next(self.model.parameters()).device

        if progress_bar:
            input_loader = tqdm(input_loader, desc="Training", unit="Batches", position=0, leave=False)

        self.model.train()

        for input, labels in input_loader:
            input, \
            labels      = input.to(device), labels.to(device)
            output      = self.model.forward(input)
            loss        = self.criterion(output, labels)
            loss_total += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        return loss_total / len(input_loader)

    def predict(self, input_loader, progress_bar=False) -> float:
        accuracy = 0
        device   = next(self.model.parameters()).device

        if progress_bar:
            input_loader = tqdm(input_loader, desc="Testing", unit="Batches", position=0, leave=False)

        self.model.eval()

        with T.inference_mode():
            for i, (input, labels) in enumerate(input_loader):
                input, \
                labels    = input.to(device), labels.to(device)
                output    = self.model.forward(input).argmax(dim=1)
                accuracy += T.sum( output == labels ).item() / len(labels)

                if progress_bar:
                    input_loader.set_description(f"Accuracy = {accuracy / (i +1):.2%}")

        return accuracy / len(input_loader)

    def test(self, input_loader, progress_bar=False) -> tuple[float, float]:
        loss_total = 0
        device     = next(self.model.parameters()).device

        self.model.eval()

        if progress_bar:
            input_loader = tqdm(input_loader, desc="Testing", unit="Batches", position=0, leave=False)

        with T.inference_mode():
            for i, (input, labels) in enumerate(input_loader):
                input, \
                labels      = input.to(device), labels.to(device)
                outputs     = self.model(input)
                loss_total += self.criterion(outputs, labels).item()

                if progress_bar:
                    input_loader.set_description(f"Loss = {loss_total / (i +1):.2e}")

        return loss_total / len(input_loader)