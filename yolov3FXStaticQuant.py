import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torchvision
from torchvision import transforms

from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.qconfig import default_qconfig

from collections import OrderedDict
from nets.yolo import YoloBody

from torch.utils.data import DataLoader
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_anchors, get_classes

from nets.darknet import darknet53

def getDataloader():
    train_annotation_path   = '2007_train.txt'
    with open(train_annotation_path) as f:
        train_lines = f.readlines()

    input_shape     = [416, 416]
    train_dataset   = YoloDataset(train_lines, input_shape, num_classes, train = True)
    gen             = DataLoader(train_dataset, shuffle = True, batch_size = 12, num_workers = 4, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)
    return gen

def calibrate(model, data_loader):
    model.eval()
    i = 1 # 总共208轮
    
    for iteration, batch in enumerate(data_loader):
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            
            images  = torch.from_numpy(images).type(torch.FloatTensor)
            targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]

            # print(images.shape)

            model(images)
            if i%10==0:
                print(i)
            i+=1

def print_size_of_model(model):
    if isinstance(model, torch.jit.RecursiveScriptModule):
        torch.jit.save(model, "temp.p")
    else:
        torch.jit.save(torch.jit.script(model), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p")/1e6)
    os.remove("temp.p")

if __name__ == "__main__":
    
    state_dict = torch.load('logs/ep099-loss2.616-val_loss4.583.pth')

    classes_path    = 'model_data/voc_classes.txt'
    anchors_path    = 'model_data/yolo_anchors.txt'
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors     = get_anchors(anchors_path)

    traindataloader = getDataloader() # 生成训练数据集
    
    # print(next(iter(traindataloader))[0])

    float_model = YoloBody(anchors_mask, num_classes, pretrained=False) # 原始模型保存一份
    float_model.load_state_dict(state_dict, strict=False)
    float_model.to('cpu')
    float_model.eval()

    model_to_quantize = YoloBody(anchors_mask, num_classes, pretrained=False) # 用来量化的模型
    model_to_quantize.load_state_dict(state_dict, strict=False)
    model_to_quantize.to('cpu')
    model_to_quantize.eval()

    qconfig = default_qconfig
    qconfig_mapping = QConfigMapping().set_global(qconfig)
    
    # example_inputs = (next(iter(traindataloader))[0])
    example_inputs = torch.randn((12, 3, 416, 416))

    prepared_model = prepare_fx(model_to_quantize, qconfig_mapping, example_inputs)
    # print(prepared_model.graph)
    calibrate(prepared_model, traindataloader)

    print("BEGIN") 
    quantized_model = convert_fx(prepared_model)
    # print(quantized_model) 
    print("OK") 

    print("Size of model before quantization")
    print_size_of_model(float_model)# 

    print("Size of model after quantization")
    print_size_of_model(quantized_model)#  

    torch.jit.save(torch.jit.script(quantized_model), 'logs/outQuant.pth')
    loaded_quantized_model = torch.jit.load('logs/outQuant.pth') 
    