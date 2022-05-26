import os,argparse,json,torch,torchmetrics
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np

from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping,StochasticWeightAveraging,LearningRateMonitor,ModelCheckpoint,RichProgressBar,DeviceStatsMonitor
from ablang.model import AbLang

from transformers import PreTrainedTokenizerFast,DataCollatorForLanguageModeling
from datasets import load_dataset

def add_spaces(example):
    example['text'] = ' '.join(example['text'])
    return example

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

class AbLang_Finetune(pl.LightningModule):
    def __init__(self,model,dataset,lr=1e-3,batch_size=64,weight_decay=0.01,model_folder='/mnt/data/AbLang/weights/heavy'):
        super().__init__()
        self.model = model
        self.weight_decay = weight_decay
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('ablang_huggingface_tokenizer/',model_max_length=155,truncation=True)
        self.AbRep = self.model.AbRep
        self.AbHead = self.model.AbHead

        self.dataset = dataset
        self.lr = lr
        self.batch_size = batch_size
        self.metric = torchmetrics.Accuracy()


    def forward(self,args,kwargs):
         return self.model(*args,**kwargs)
    
    def training_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        labels = batch['labels']
        mask = batch['attention_mask'].bool()
        logits = self.model(input_ids,attention_mask=mask)
        predictions  = torch.argmax(logits,2)[labels!=-100]
        targets = labels[labels!=-100]
        acc = self.metric(predictions,targets)
        loss = F.cross_entropy(logits.transpose(1,2),labels,ignore_index=-100)
        self.log('train_loss',loss)
        self.log('train_acc',acc)
        return loss

    def validation_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        ## These are the correct labels for the loss
        labels = batch['labels']
        mask = batch['attention_mask'].bool()
        logits = self.model(input_ids,attention_mask=mask)
        predictions  = torch.argmax(logits,2)[labels!=-100]
        targets = labels[labels!=-100]
        acc = self.metric(predictions,targets)
        loss = F.cross_entropy(logits.transpose(1,2),labels,ignore_index=-100)
        self.log('val_acc',acc)
        self.log('val_loss',loss)
        return {"loss": loss, "acc": acc, "preds": predictions, "labels": labels}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(),lr=self.lr,weight_decay=self.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=self.lr,epochs=100,div_factor=100,steps_per_epoch=1458878,cycle_momentum=False,verbose=True)
        # scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=2, max_iters=120)
        
        return [optimizer], [scheduler]
    def train_dataloader(self):
        return DataLoader(self.dataset['train'].with_format('torch'),batch_size=self.batch_size,
                          collate_fn=DataCollatorForLanguageModeling(tokenizer=self.tokenizer),pin_memory=True)
    def val_dataloader(self):
        return DataLoader(self.dataset['val'].with_format('torch'),batch_size=self.batch_size,
                          collate_fn=DataCollatorForLanguageModeling(tokenizer=self.tokenizer),pin_memory=True)
    def test_dataloader(self):
        return DataLoader(self.dataset['test'].with_format('torch'),batch_size=self.batch_size,
                          collate_fn=DataCollatorForLanguageModeling(tokenizer=self.tokenizer),pin_memory=True)
        
if __name__ == '__main__':
    pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    ## Load Dataset Dictionary
    data_path = '/human_oas/'
    hf_tokenizer = PreTrainedTokenizerFast.from_pretrained('ablang_huggingface_tokenizer/',model_max_length=155,truncation=True)

    train_dataset = load_dataset('text',data_files=os.path.join(data_path,f'human_heavy_oas_filtered_len_train.txt.gz'),
                                   streaming=True)['train']
    train_dataset = train_dataset.map(add_spaces)
    train_dataset = train_dataset.map(lambda x: hf_tokenizer(x['text'],return_token_type_ids=False,return_special_tokens_mask=True,
                                                        return_tensors='pt',padding=True),remove_columns=['text'],batched=True,batch_size=100000)
    train_dataset.shuffle(buffer_size=50000)

    val_dataset = load_dataset('text',data_files=os.path.join(data_path,f'human_heavy_oas_filtered_len_val.txt.gz'),streaming=True)['train']
    val_dataset = val_dataset.map(add_spaces)
    val_dataset = val_dataset.map(lambda x: hf_tokenizer(x['text'],return_token_type_ids=False,return_special_tokens_mask=True,
                                                        return_tensors='pt',padding=True),remove_columns=['text'],batched=True,batch_size=100000)
    val_dataset.shuffle(buffer_size=50000)

    test_dataset = load_dataset('text',data_files=os.path.join(data_path,f'human_heavy_oas_filtered_len_test.txt.gz'),streaming=True)['train']
    test_dataset = test_dataset.map(add_spaces)
    test_dataset = test_dataset.map(lambda x: hf_tokenizer(x['text'],return_token_type_ids=False,return_special_tokens_mask=True,
                                                        return_tensors='pt',padding=True),remove_columns=['text'],batched=True,batch_size=100000)
    test_dataset.shuffle(buffer_size=50000)


    dataset = {'train':train_dataset,'val':val_dataset,'test':test_dataset}

    learning_rate_monitor = LearningRateMonitor(logging_interval='step')

    epoch_checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor="val_acc",
        mode="max",
        dirpath="ablang_human_heavy/",
        filename="Ablang-human-heavy-{epoch:02d}-{val_acc:.2f}",verbose=True
    )

    logger = TensorBoardLogger("Ablang_Human", name="Ablang_Human_Heavy")
    trainer = pl.Trainer(gpus=1,benchmark=True,logger=logger, gradient_clip_val=0.005,
                        profiler="simple",accelerator="auto",auto_select_gpus=True,max_epochs=120,precision=16,
                        callbacks = [epoch_checkpoint_callback, StochasticWeightAveraging(), 
                                      RichProgressBar(),  DeviceStatsMonitor(), learning_rate_monitor])
    
    
    model_folder = 'weights/heavy'
    hparams_file = os.path.join(model_folder, 'hparams.json')
    model_file = os.path.join(model_folder, 'amodel.pt')
    with open(hparams_file, 'r', encoding='utf-8') as f:
        hparams = argparse.Namespace(**json.load(f))    
    ablang_model = AbLang(hparams)
    ablang_model.load_state_dict(torch.load(model_file,map_location='cpu'))

    model = AbLang_Finetune(ablang_model,dataset)

    # batch_finder = trainer.tuner.scale_batch_size(model,mode='binsearch')
    # print(batch_finder)
    model.batch_size = 128
    model.model.batch_size = 128

    lr_finder = trainer.tuner.lr_find(model)
    print(lr_finder.suggestion())
    model.lr = lr_finder.suggestion()
    model.model.lr = lr_finder.suggestion()
    torch.cuda.empty_cache()
    trainer.fit(model)