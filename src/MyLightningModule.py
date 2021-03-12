import pytorch_lightning as pl
import torch
import numpy as np

class LightningModule(pl.LightningModule):

    def __init__(self, model, hparams=None):
        super().__init__()
        self.model = model
        self.loss = torch.nn.CrossEntropyLoss()
        
    def training_step(self, batch, batch_idx):
        x, y = batch
    
        scores = self.model(x)        
        loss = self.loss(scores, y)
        self.log("training_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        scores = self.model(x)
        predictions = torch.argmax(scores, dim=1)

        num_true = torch.sum(predictions == y)
        num_false = torch.sum(predictions != y)

        return num_true.item(), num_false.item()


    def validation_epoch_end(self, validation_step_outputs):
        validation_step_outputs = np.array(validation_step_outputs)
        total = reduce(validation_step_outputs, "b tf -> tf", reduction=sum)
        acc = total[0] / (total[0] + total[1])
        self.log("validation_accuracy", acc, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=2)
        return {'optimizer': optimizer, 
                'lr_scheduler': scheduler, 
                'monitor': 'training_loss'}
