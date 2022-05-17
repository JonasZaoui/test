import monai 
import matplotlib.pyplot as plt
import torchio as tio
import time
import pandas as pd
from datetime import datetime
import torch 
import torchvision
import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import random_split, DataLoader
import os


plt.rcParams['figure.figsize'] = 12, 8
monai.utils.set_determinism()

print('Last run on', time.ctime())


print(tio.__version__)
print(torch.__version__)
print(torchvision.__version__)
print(monai.__version__)
print(pl.__version__)


class Dataset(pl.LightningDataModule):
    def __init__(self,dataset_dir,batch_size,train_val_ratio):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_dir = Path(dataset_dir)
        self.subjects = None
        self.train_val_ratio = train_val_ratio
        self.test_subjects = None
        self.preprocess = None
        self.transform = None
        self.train_set = None
        self.val_set = None
        self.test_set = None

    
    def download_data(self):
       
        def get_niis(d):
            return sorted(p for p in d.glob('*.nii*') if not p.name.startswith('.'))

        image_training_paths = get_niis(self.dataset_dir / 'mri')
        label_training_paths = get_niis(self.dataset_dir / 'labels')
        return image_training_paths, label_training_paths

    def prepare_data(self):
        image_training_paths, label_training_paths = self.download_data()

        self.subjects = []
        for image_path, label_path in sorted(zip(image_training_paths, label_training_paths)):
            subject = tio.Subject(
                image=tio.ScalarImage(image_path),
                label=tio.LabelMap(label_path)
            )
            self.subjects.append(subject)

        
    
    def get_preprocessing_transform(self):
        preprocess = tio.Compose([
            tio.RescaleIntensity((-1, 1)),
            tio.Pad((64,64,16)),
        ])
        return preprocess


    # def get_preprocessing_transform(self):
    #     preprocess = tio.RescaleIntensity((-1, 1))
    #     return preprocess
        
    def get_augmentation_transform(self):
        augment = tio.RandomGamma(p=0.5)
        return augment

    def setup(self,stage=None):
        
        num_subjects = len(self.subjects)
        num_train_subjects = int(round(num_subjects * self.train_val_ratio))
        num_val_subjects = num_subjects - num_train_subjects
        splits = num_train_subjects, num_val_subjects
        train_subjects, val_subjects = random_split(self.subjects, splits)

        self.preprocess = self.get_preprocessing_transform()
        augment = self.get_augmentation_transform()
        self.transform = tio.Compose([self.preprocess, augment])

        self.train_set = tio.SubjectsDataset(train_subjects, transform=self.transform) #data augmentation only on the trainset. 
        self.val_set = tio.SubjectsDataset(val_subjects, transform=self.preprocess)


        

    def train_dataloader(self):
        patches_queue_train = tio.Queue(self.train_set,max_length=60,samples_per_volume=10,sampler=tio.data.LabelSampler(patch_size=(128,128,32)),num_workers=2) 
        patches_loader_train = DataLoader(patches_queue_train,self.batch_size,num_workers=0) 

        return patches_loader_train


    def val_dataloader(self):
        patches_queue_val = tio.Queue(self.val_set,max_length=60,samples_per_volume=10,sampler=tio.data.LabelSampler(patch_size=(128,128,32)),num_workers=2) 
        patches_loader_val = DataLoader(patches_queue_val,self.batch_size,num_workers=0) 

  

        return patches_loader_val


data = Dataset(
    dataset_dir = '/network/lustre/iss02/cati/collabs/cmbs/training_jonas/train_positif',
    batch_size=4, 
    train_val_ratio=0.8
)


data.prepare_data()
data.setup()
print('Training:  ', len(data.train_set))
print('Validation: ', len(data.val_set))


class Model(pl.LightningModule):
    def __init__(self, net, criterion, learning_rate, optimizer_class):
        super().__init__()
        self.lr = learning_rate #pas d'apprentissage pour optimiseur
        self.net = net #model utilisé 
        self.criterion = criterion #fonction cout a minimiser
        self.optimizer_class = optimizer_class #Optimiseur choisi
    
    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr) #On applique l'optimiseur aux paramètres du modele
        return optimizer
    
    def prepare_batch(self, batch):
        return batch['image'][tio.DATA], batch['label'][tio.DATA]
    
    def infer_batch(self, batch): # calcul l'erreur entre la sortie du model et le label. 
        x, y = self.prepare_batch(batch) 
        y_hat = self.net(x) #prediction
        return y_hat, y

    def training_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch) # recupere la prediction et le label
        loss = self.criterion(y_hat, y) #calcul la perte entre les 2
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
        
    def validation_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return loss




#les patchs sont en 144x144x32
#les images sont en 208x208x48


tb_logger = pl.loggers.TensorBoardLogger(save_dir="/network/lustre/iss02/cati/collabs/cmbs/training_jonas/log")


unet = monai.networks.nets.UNet(
    dimensions=3,
    in_channels=1,
    out_channels=1, 
    channels=(8, 16, 32, 64),
    strides=(2, 2, 2),
)


model = Model(
    net=unet,
    criterion=monai.losses.GeneralizedDiceLoss(sigmoid=True),
    learning_rate=1e-3,
    optimizer_class=torch.optim.AdamW,
)



# early_stopping = pl.callbacks.early_stopping.EarlyStopping(
#     monitor='val_loss',
# )

# trainer = pl.Trainer( 
#     gpus=1, 
#     precision=16, #!!!!!!!!!!!
#     callbacks=[early_stopping],
# )

trainer = pl.Trainer( 
    gpus=1, 
    precision=16,max_epochs=200, #!!!!!!!!!!!
    logger=tb_logger
)

trainer.logger._default_hp_metric = False


start = datetime.now()
print('Training started at', start)
trainer.fit(model=model, datamodule=data)
print('Training duration:', datetime.now() - start)



torch.save(model.state_dict(), '/network/lustre/iss02/cati/collabs/cmbs/training_jonas/model_weights.pth')
