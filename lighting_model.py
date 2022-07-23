"""
Implementation of Yolo (v1) architecture
with slight modification with added BatchNorm.
"""
from gc import callbacks
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from dataset import VOCDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from loss import YoloLoss
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)

writer=SummaryWriter("runs/cifar10")
""" 
Information about architecture config:
Tuple is structured by (kernel_size, filters, stride, padding) 
"M" is simply maxpooling with stride 2x2 and kernel 2x2
List is structured by tuples and lastly int with number of repeats
"""


seed = 123
torch.manual_seed(seed)

# Hyperparameters etc. 
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 8 # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0
EPOCHS = 10
NUM_WORKERS = 4
PIN_MEMORY = True
LOAD_MODEL = True
FILE_NAME="best"
LOAD_MODEL_DIR = "./checkpoints"
IMG_DIR = "../../../data/images"
LABEL_DIR = "../../../data/labels"
loss_fn = YoloLoss()
mean_loss = []

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])

train_dataset = VOCDataset(
    "../../../data/100examples.csv",
    transform=transform,
    img_dir=IMG_DIR,
    label_dir=LABEL_DIR,
)

# use 20% of training data for validation
train_set_size = int(len(train_dataset) * 1)
valid_set_size = len(train_dataset) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = random_split(train_dataset, [train_set_size, valid_set_size], generator=seed)

train_loader = DataLoader(
    dataset=train_set,
    batch_size=BATCH_SIZE,
    
    pin_memory=PIN_MEMORY,
    shuffle=True,
    drop_last=True,
)
val_loader = DataLoader(
    dataset=valid_set,
    batch_size=BATCH_SIZE,
    
    pin_memory=PIN_MEMORY,
    shuffle=False,
    drop_last=True,
)


test_dataset = VOCDataset(
    "../../../data/100examples.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    
    pin_memory=PIN_MEMORY,
    shuffle=False,
    drop_last=True,
)

class CNNBlock(pl.LightningModule):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

    
class LitYolov1(pl.LightningModule):
    def __init__(self, in_channels=3, **kwargs):
        super(LitYolov1, self).__init__()
        architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))
   
    
    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                ]
                in_channels = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes

        # In original paper this should be
        # nn.Linear(1024*S*S, 4096),
        # nn.LeakyReLU(0.1),
        # nn.Linear(4096, S*S*(B*5+C))

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 2048),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(2048, S * S * (C + B * 5)),
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)

        
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        avg_mean_loss=sum(mean_loss)/len(mean_loss)
        self.log("Train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("Train_avg_mean_loss", avg_mean_loss)
        return {"loss":loss, "avg_mean_loss" : avg_mean_loss}

    def training_step_end(self, training_step_outputs):
        return training_step_outputs

    def training_epoch_end(self, training_step_outputs):

        pred_boxes, target_boxes = get_bboxes(train_loader, self, iou_threshold=0.5, threshold=0.4 )
        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint" )
        avg_mean_loss=training_step_outputs[1]
        print(f"\nMean loss (Train) was {avg_mean_loss}")
        print(f"Train mAP: {mean_avg_prec}", f"Epoch: {self.current_epoch}\n")
        self.log("Train mAP", mean_avg_prec)



    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = loss_fn(preds, y)
        mean_loss.append(loss.item())
        avg_mean_loss=sum(mean_loss)/len(mean_loss)
        self.log("Validation Loss : ", loss)
        self.log("Validation_avg_mean_loss", avg_mean_loss)
        return {"loss":loss, "avg_mean_loss" : avg_mean_loss}

    def validation_epoch_end(self,validating_step_outputs):
        #print(len(validating_step_outputs)) ## This will be same as number of validation batches
        pred_boxes, target_boxes = get_bboxes(val_loader, self, iou_threshold=0.5, threshold=0.4)
        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")
        avg_mean_loss=validating_step_outputs[1]
        print(f"\nMean loss (Validation) was {avg_mean_loss}")
        print(f"Validation mAP: {mean_avg_prec}", f"Epoch: {self.current_epoch}\n")
        self.log("Validation mAP", mean_avg_prec)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)

        for idx in range(8):
               bboxes = cellboxes_to_boxes(preds)
               bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
               plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)

        import sys
        sys.exit()
        loss = loss_fn(preds, y)
        mean_loss.append(loss.item())
        avg_mean_loss=sum(mean_loss)/len(mean_loss)
        self.log("Test Loss : ", loss)
        self.log("Test_avg_mean_loss", avg_mean_loss)

        return {"loss":loss, "avg_mean_loss" : avg_mean_loss}

    def test_epoch_end(self,validating_step_outputs):
        #print(len(validating_step_outputs)) ## This will be same as number of validation batches
        pred_boxes, target_boxes = get_bboxes(test_loader, self, iou_threshold=0.5, threshold=0.4)
        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")
        avg_mean_loss=validating_step_outputs[1]
        print(f"\nMean loss (Test) was {avg_mean_loss}")
        print(f"Test mAP: {mean_avg_prec}", f"Epoch: {self.current_epoch}\n")
        self.log("Test mAP", mean_avg_prec)
        

    def predict_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        return preds
    
    def configure_optimizers(self):
        #optimizer = optim.Adam(self.parameters(), lr=1e-3)
        optimizer = optim.Adam(
            self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        return optimizer
    
  

if __name__=='__main__':
    checkpoint_callback = ModelCheckpoint(
        monitor="Train mAP",
        dirpath=LOAD_MODEL_DIR,
        
        save_top_k=1,        # save the best model
        mode="min",
        every_n_epochs=1,
        save_last=True,
        auto_insert_metric_name=True,
        filename=FILE_NAME,

    )
      
    if not LOAD_MODEL:
        trainer = Trainer(max_epochs=EPOCHS,
            enable_checkpointing=True,
            gpus=1,
            default_root_dir=LOAD_MODEL_DIR,
            checkpoint_callback=True,
            callbacks=[checkpoint_callback],
            enable_progress_bar=True,
            enable_model_summary=True,
            logger = True, 
            num_sanity_val_steps=0,
            
            )
        model = LitYolov1(split_size=7, num_boxes=2, num_classes=20)
    else:
        checkpoint_callback = ModelCheckpoint(dirpath=LOAD_MODEL_DIR,filename=FILE_NAME,monitor="Train mAP",save_top_k=1)
        trainer = Trainer(callbacks=[checkpoint_callback],gpus=1,limit_train_batches=0,limit_val_batches=0)
        model = LitYolov1.load_from_checkpoint(checkpoint_path=os.path.join(os.path.join(LOAD_MODEL_DIR,'last' + '.ckpt')),split_size=7, num_boxes=2, num_classes=20)
    
    examples = iter(train_loader)
    example_data, example_targets = examples.next()
    writer.add_graph(model, example_data)
    del examples, example_data, example_targets, writer # To free some RAM space
    
    # trainer.fit(model,train_dataloaders=train_loader,val_dataloaders=val_loader)
    trainer.fit(model,train_dataloaders=train_loader)
    trainer.test(model=model, dataloaders=test_loader)

