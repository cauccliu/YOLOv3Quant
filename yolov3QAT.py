import torch
from torch.ao.quantization import (
  get_default_qconfig_mapping,
  get_default_qat_qconfig_mapping,
  QConfigMapping,
)
import torch.ao.quantization.quantize_fx as quantize_fx

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.yolo import YoloBody
from nets.yolo_training import YOLOLoss, weights_init
from utils.callbacks import LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_anchors, get_classes
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":

    Cuda            = True
    classes_path    = 'model_data/voc_classes.txt'
    
    anchors_path    = 'model_data/yolo_anchors.txt'
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    
    model_path      = ''
   
    input_shape     = [416, 416]
    pretrained      = False
    
    Init_Epoch          = 0
    Freeze_Epoch        = 0
    Freeze_batch_size   = 8
    Freeze_lr           = 1e-3
    
    UnFreeze_Epoch      = 1
    Unfreeze_batch_size = 4
    Unfreeze_lr         = 1e-4
    
    Freeze_Train        = True
   
    num_workers         = 4
    
    train_annotation_path   = '2007_train.txt'
    val_annotation_path     = '2007_val.txt'

    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors     = get_anchors(anchors_path)

    model_to_quantize  = YoloBody(anchors_mask, num_classes, pretrained=pretrained)
    model_to_quantize = model_to_quantize.to('cuda')

    # 量化感知训练
    model_to_quantize .eval()
    qconfig_mapping = get_default_qat_qconfig_mapping("qnnpack")
    model_to_quantize.train()
    example_inputs = torch.randn((12, 3, 416, 416))

    model_prepared = quantize_fx.prepare_qat_fx(model_to_quantize, qconfig_mapping, example_inputs)

    model_train = model_prepared.train()
    model_train = model_train.cuda()

    yolo_loss    = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask)
    loss_history = LossHistory("logs/")

    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
    
    if True:
        batch_size  = Freeze_batch_size
        lr          = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch   = Freeze_Epoch
                        
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
        
        optimizer       = optim.Adam(model_train.parameters(), lr, weight_decay = 5e-4)
        lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset   = YoloDataset(train_lines, input_shape, num_classes, train = True)
        val_dataset     = YoloDataset(val_lines, input_shape, num_classes, train = False)
        gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate)

        if Freeze_Train:
            for param in model_prepared.backbone.parameters():
                param.requires_grad = False

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_train, model_prepared, yolo_loss, loss_history, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
            lr_scheduler.step()
            
    if True:
        batch_size  = Unfreeze_batch_size
        lr          = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch   = UnFreeze_Epoch
                        
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
        
        optimizer       = optim.Adam(model_train.parameters(), lr, weight_decay = 5e-4)
        lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset   = YoloDataset(train_lines, input_shape, num_classes, train = True)
        val_dataset     = YoloDataset(val_lines, input_shape, num_classes, train = False)
        gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate)

        if Freeze_Train:
            for param in model_prepared.backbone.parameters():
                param.requires_grad = True

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_train, model_prepared, yolo_loss, loss_history, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
            lr_scheduler.step()

    model_quantized = quantize_fx.convert_fx(model_prepared)
    torch.jit.save(torch.jit.script(model_quantized), 'logs/outQATQuant.pth')
    loaded_quantized_model = torch.jit.load('logs/outQATQuant.pth')

    """ 
    model_fused = quantize_fx.fuse_fx(model_quantized)
    torch.jit.save(torch.jit.script(model_fused), 'logs/outQATQuantfused.pth')
    loaded_quantized_model = torch.jit.load('logs/outQATQuantfused.pth') """
    
