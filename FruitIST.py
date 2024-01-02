import os
import torch as T
import torchvision as TV
from math import prod
from tqdm.auto import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from safetensors.torch import load_model, save_model
from Trainer import TrainerV1


## Hyperparameters
BATCH_SIZE      = 64
NUM_EPOCHS      = 100
LEARNING_RATE   = 1e-4
LEARNING_STEP   = 5
LEARNING_GAMMA  = 1e-1

TIME_FORMAT     = "%H:%M:%S"
CHECKPOINT_PATH = "fruitist_state.safetensors"


## Check for CUDA
device    = T.device("cuda" if T.cuda.is_available() else "cpu")

## Load data
#### Create transform
transform = TV.transforms.Compose((
    TV.transforms.ToTensor(),
))

#### Load datasets
training_set = TV.datasets.ImageFolder("./Training/", transform=transform)
testing_set  = TV.datasets.ImageFolder("./Testing/", transform=transform)

#### Create data loaders
train_loader = DataLoader(dataset=training_set,
                          batch_size=BATCH_SIZE,
                          shuffle=True)
test_loader  = DataLoader(dataset=testing_set,
                          batch_size=BATCH_SIZE,
                          shuffle=False)

#### Get data information
num_channels = training_set[0][0].shape[0]
num_classes  = len(training_set.classes)
image_size   = prod(training_set[0][0].shape[1:])

## Model
#### Create model
model = TV.models.resnet18(num_classes=num_classes).to(device)

#### Create utilities
criterion = T.nn.CrossEntropyLoss()
optimizer = T.optim.Adam(params=model.parameters(),
                         lr=LEARNING_RATE)
scheduler = T.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                        gamma=LEARNING_GAMMA,
                                        step_size=LEARNING_STEP)

#### Create trainer
trainer = TrainerV1(model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler)


## Train
if __name__ == "__main__":
    best = float("INF")

    print(f"Device is {device}")

    print(f"Loaded training set with {len(training_set)} images")
    print(f"Loaded testing set with {len(testing_set)} images")
    print(f"Batch size is {BATCH_SIZE}\n")
    
    print(f"{num_channels = }")
    print(f"{num_classes  = }")
    print(f"{image_size   = }")
    print("\n")
    
    #### Check for saved state
    if os.path.isfile(CHECKPOINT_PATH):
        print("Saved state detected. Loading file . . . ")

        try:
            load_model(model, CHECKPOINT_PATH)
        except:
            print(f"Unable to load saved state. Starting over")
        else:
            best     = trainer.test(test_loader,   progress_bar=True)
            accuracy = trainer.predict(test_loader, progress_bar=True)

            print(f"Loaded saved state with a loss of {best:.2e} accuracy of {accuracy:.2%}")
    else:
        print("No saved state file detected")


    print(f"[{ datetime.now().strftime(TIME_FORMAT) }] Starting Training . . . ")

    #### Learn
    for epoch in tqdm(range(NUM_EPOCHS), desc="Training", unit="Epoch"):
        saved    = ""
        time     = datetime.now()
        loss     = trainer.forward(train_loader, progress_bar=True)
        accuracy = trainer.predict(test_loader,  progress_bar=True)
        
        if accuracy >= best:
            best  = accuracy
            saved = " | Saved"

            save_model(model, CHECKPOINT_PATH)

        print(f"[{datetime.now().strftime(TIME_FORMAT)}] (+{(datetime.now() - time).total_seconds():.0f}s) Epoch: {epoch +1}/{NUM_EPOCHS} | Loss = {loss:.2e} | Accuracy = {accuracy:.2%}{saved}")
