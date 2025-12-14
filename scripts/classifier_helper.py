#this software is provided as is, and is not guaranteed to work or be suitable for any particular purpose
#for use only in the context of the "Machine Aesthetics" course at GSD. Do not share or distribute
#copyright 2023-2024 Panagiotis Michalatos : pan.michalatos@gmail.com

from collections import Counter
from enum import Enum
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import WeightedRandomSampler
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
#from torch.utils.tensorboard import SummaryWriter
import os
from PIL import Image
import cv2
import numpy as np
import time
import json

import logging

import random

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

logging.getLogger("PIL.PngImagePlugin").setLevel(logging.CRITICAL + 1)
logging.getLogger("PIL.TiffImagePlugin").setLevel(logging.CRITICAL + 1)

def set_parameter_requires_grad(model, freeze_pretrained_parameters):
    if freeze_pretrained_parameters:
        for param in model.parameters():
            param.requires_grad = False


def createClassifierModel(model_name, num_classes, freeze_pretrained_parameters, use_pretrained=True) -> nn.Module:
    model_ft = None

    if model_name == "resnet":
        """ Resnet18
        """

        if use_pretrained:
            model_ft = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            model_ft = models.resnet18()

        set_parameter_requires_grad(model_ft, freeze_pretrained_parameters)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "alexnet":
        """ Alexnet
        """

        if use_pretrained:
            model_ft = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        else:
            model_ft = models.alexnet()

        set_parameter_requires_grad(model_ft, freeze_pretrained_parameters)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
    elif model_name == "vgg":
        """ VGG11_bn
        """

        if use_pretrained:
            model_ft = models.vgg11_bn(weights=models.VGG11_BN_Weights.DEFAULT)
        else:
            model_ft = models.vgg11_bn()
            
        set_parameter_requires_grad(model_ft, freeze_pretrained_parameters)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
    elif model_name == "squeezenet":
        """ Squeezenet
        """

        if use_pretrained:
            model_ft = models.squeezenet1_0(weights=models.SqueezeNet_Weights.DEFAULT)
        else:
            model_ft = models.squeezenet1_0()
            
        set_parameter_requires_grad(model_ft, freeze_pretrained_parameters)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
    elif model_name == "densenet":
        """ Densenet
        """

        if use_pretrained:
            model_ft = models.densenet121(weights=models.DenseNet_Weights.DEFAULT)
        else:
            model_ft = models.densenet121()


        set_parameter_requires_grad(model_ft, freeze_pretrained_parameters)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
    elif model_name == "vit_b_32":
        """ Vision Transformer B_32 (ViT-B_32) """
        if use_pretrained:
            model_ft = models.vit_b_32(weights=models.ViT_B_32_Weights.DEFAULT)
        else:
            model_ft = models.vit_b_32()

        set_parameter_requires_grad(model_ft, freeze_pretrained_parameters)
        num_ftrs = model_ft.heads.head.in_features
        model_ft.heads.head = nn.Linear(num_ftrs, num_classes)
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft

def getAllImagesFromFolder(folder):
    image_files = os.listdir(folder)
    image_files = [f for f in image_files if f.endswith('.png') or f.endswith('.jpg')]
    return image_files

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

IMAGENET_NORMALIZATION = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

IMAGENET_TRANSFORM = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])  

CLIP_TRANSFORM = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD)
        ])

class JITModel:
    def __init__(self, model_path : str, inp_shape : list[int], dtype , device : torch.device):

        self.device = device

        self.model_path = model_path
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

        self.inp_shape : list[int] = inp_shape
        
        #run once to get output shape
        inp_tensor = torch.ones(inp_shape, dtype=dtype, device=self.device)
        output : torch.Tensor = self.model(inp_tensor)
        self.out_shape : list[int]  = output.shape

    def run(self, inp_tensor : torch.Tensor):
        inp_tensor = inp_tensor.to(self.device)
        output : torch.Tensor = self.model(inp_tensor)
        return output


class VisualModel(JITModel):
    def __init__(self, model_path, transforms, device : torch.device):
        super().__init__(model_path, [1, 3, 224, 224], torch.float32, device)
        self.transforms = transforms
    
    def classify(self, image):
        #if image is opencv image, convert to PIL
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        with torch.no_grad():
            image_tensor : torch.Tensor = self.transforms(image)
            image_tensor = image_tensor.unsqueeze(0)

            outputs = self.run(image_tensor)

        return outputs
    
    def classifyAllImagesInFolder(self, folder, max_count = -1) -> list[tuple[str, np.ndarray]]:
        image_files = getAllImagesFromFolder(folder)
        image_files = [os.path.join(folder, f) for f in image_files]

        if max_count > 0 and max_count < len(image_files):
            image_files = random.sample(image_files, max_count)

        results = []
        for i, image_file in enumerate(image_files):
            print(f'processing image {i}/{len(image_files)}')
            image = Image.open(image_file).convert('RGB')
            outputs = self.classify(image)
            results.append((image_file, outputs[0].cpu().numpy()))

        return results


class ImageNetModel(VisualModel):
    def __init__(self, model_path, device : torch.device):
        super().__init__(model_path, IMAGENET_TRANSFORM, device)

class ClipVisualModel(VisualModel):
    def __init__(self, model_path, device : torch.device):
        super().__init__(model_path, CLIP_TRANSFORM, device)

class CLIPTextualModel(JITModel):
    def __init__(self, model_path, device : torch.device):
        super().__init__(model_path, [1, 77], torch.int32, device)

    def classify(self, text):        
        with torch.no_grad():
            text_tensor : torch.Tensor = clip.tokenize(text)
            text_tensor = text_tensor.to(self.device)

            outputs = self.run(text_tensor)

        return outputs

class Classifier:
    AVAILABLE_MODEL_TYPES = ["resnet", "alexnet", "vgg", "squeezenet", "densenet", "vit_b_32"]

    @staticmethod
    def loadFromFolder(
        model_folder : str,
        device : torch.device
    ) -> 'Classifier':
        classifier = Classifier(model_folder, device)
        classifier.load()
        return classifier
       

    @staticmethod
    def createWithImgNetModel(
        model_folder : str,       
        device : torch.device, 
        model_type : str = 'vit_b_32'             
    ) -> 'Classifier':
        classifier = Classifier(model_folder, device)
        classifier.model_type = model_type
        return classifier
    
    @staticmethod
    def createBinaryClassifier(
        model_folder : str,       
        device : torch.device, 
        model_type : str = 'resnet'             
    ) -> 'Classifier':
        """
        Create a binary classifier with enforced label mapping: real=0, game=1.
        This ensures consistent label ordering regardless of filesystem folder order.
        """
        classifier = Classifier(model_folder, device)
        classifier.model_type = model_type
        # Enforce fixed binary labels
        classifier.class_to_idx = {"real": 0, "game": 1}
        classifier.updateClassNames()
        return classifier
    

    def __init__(self, model_folder : str, device: torch.device):
        self.device = device

        self.model_folder : str = model_folder
        self.model_type : str = 'vit_b_32'

        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)

        self.weights_file : str = os.path.join(self.model_folder, 'model_weights.pth')
        self.json_file : str = os.path.join(self.model_folder, 'model_info.json')

        self.image_size : int = 224
        self.image_channels : int = 3

        self.class_to_idx : dict[str, int] = {}
        self.class_names : list[str] = []

        self.model : nn.Module = None
        self.freeze_pretrained_parameters : bool = True
        self.training_loss : float = 0.0
        self.training_accuracy : float = 0.0
       
        self.inference_transforms = IMAGENET_TRANSFORM

    @property
    def class_count(self):
        return len(self.class_to_idx)
    
    def updateClassNames(self):
        self.class_names = [None]*len(self.class_to_idx)
        for name, idx in self.class_to_idx.items():
            self.class_names[idx] = name
    
    def ensureModel(self, load_weights):
    
        if self.model is None:
            with torch.no_grad():
                self.model = createClassifierModel(self.model_type, self.class_count, freeze_pretrained_parameters = self.freeze_pretrained_parameters, use_pretrained = True)
                self.model.to(self.device)

                if load_weights and os.path.exists(self.weights_file):
                    self.model.load_state_dict(torch.load(self.weights_file))

        return self.model
    
    def createTrainingTransforms(self, pre_transforms : list[transforms.Normalize]):
        if pre_transforms is None:
            pre_transforms = []

        pre_transforms.append(transforms.ToTensor())
        pre_transforms.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

        return transforms.Compose(pre_transforms)
    
    def getTrainingParams(self):
        params_to_update = []
        print("Params to learn:")
        if self.freeze_pretrained_parameters:
            for name,param in self.model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t",name)
        else:
            params_to_update = self.model.parameters()
            for name,param in self.model.named_parameters():
                if param.requires_grad == True:
                    print("\t",name)

        return params_to_update
    
    def train(self, 
              epochs : int, 
              images_folder : str, 
              continue_training : bool, 
              pre_transforms : list[transforms.Normalize], 
              batch_size  : int = 4,
              learning_rate : float = 0.001,
              save_every : int = 10,
              num_workers : int = 0,
              max_images : int | None = None
        ):

        #data
        data_transforms = self.createTrainingTransforms(pre_transforms)
        image_dataset = datasets.ImageFolder(images_folder, data_transforms)
        
        # If class_to_idx is already set (e.g., from createBinaryClassifier), enforce it
        if self.class_to_idx:
            # Remap dataset labels to match our fixed mapping
            print(f"Enforcing fixed label mapping: {self.class_to_idx}")
            dataset_class_to_idx = image_dataset.class_to_idx
            
            # Verify all expected classes exist
            for class_name in self.class_to_idx.keys():
                if class_name not in dataset_class_to_idx:
                    raise ValueError(f"Expected class '{class_name}' not found in dataset. Found: {list(dataset_class_to_idx.keys())}")
            
            # Create remapping
            old_to_new = {dataset_class_to_idx[name]: self.class_to_idx[name] for name in self.class_to_idx.keys()}
            
            # Remap all targets
            image_dataset.targets = [old_to_new[t] for t in image_dataset.targets]
            image_dataset.samples = [(path, old_to_new[label]) for path, label in image_dataset.samples]
            image_dataset.class_to_idx = self.class_to_idx
        else:
            self.class_to_idx = image_dataset.class_to_idx
        
        self.updateClassNames()
        dataset_for_loader = image_dataset

        if max_images is not None and max_images > 0:
            target_count = max_images
        else:
            target_count = len(dataset_for_loader)

        targets = dataset_for_loader.targets if hasattr(dataset_for_loader, "targets") else [sample[1] for sample in dataset_for_loader.samples]
        class_counts = Counter(targets)
        sample_weights = [1.0 / class_counts[label] for label in targets]
        sampler = WeightedRandomSampler(sample_weights, num_samples=target_count, replacement=True)

        dataloader = torch.utils.data.DataLoader(
            dataset_for_loader,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False,
        )
                
        print(f'Found {len(self.class_to_idx)} classes: {self.class_to_idx}')
        
        #model
        self.ensureModel(continue_training)
        self.model.train()

        #optimizer
        params_to_update = self.getTrainingParams()
        optimizer = optim.SGD(params_to_update, lr=learning_rate, momentum=0.9)

        # Setup the loss function
        criterion = nn.CrossEntropyLoss()

                    
        # Training loop
        since = time.time()
        #writer = SummaryWriter(self.model_folder)    

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(True):
                    outputs = self.model(inputs)
                    loss : torch.Tensor = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            self.training_loss = running_loss / len(dataloader.dataset)
            self.training_accuracy = running_corrects.item() / len(dataloader.dataset)

            #tensorboard logging
            #writer.add_scalar('Loss/train', epoch_loss, epoch)
            #writer.add_scalar('Accuracy/train', epoch_acc, epoch)

            print('Loss: {:.4f} Acc: {:.4f}'.format(self.training_loss, self.training_accuracy))
            print()

            if (epoch+1) % save_every == 0:
                self.save()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        #writer.flush()
        self.save()

    def ensureBakFile(self, file_path):        
        if os.path.exists(file_path):
            bak_name = file_path + '.bak'
            if os.path.exists(bak_name):
                os.remove(bak_name)
            os.rename(file_path, file_path + '.bak')

    def save(self):
        self.ensureBakFile(self.weights_file)
        torch.save(self.model.state_dict(), self.weights_file)

        json_data = {
            'model_type' : self.model_type,
            'class_to_idx' : self.class_to_idx,
            'training_loss' : self.training_loss,
            'training_accuracy' : self.training_accuracy
        }
        
        self.ensureBakFile(self.json_file)
        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f)

    def load(self):
        if not os.path.exists(self.json_file):
            raise Exception(f'Cannot load model info from {self.json_file}')
        
        if not os.path.exists(self.weights_file):
            raise Exception(f'Cannot load model weights from {self.weights_file}')
        
        with open(self.json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

            self.model_type = json_data['model_type']
            self.class_to_idx = json_data['class_to_idx']
            self.training_loss = json_data['training_loss']
            self.training_accuracy = json_data['training_accuracy']

        self.updateClassNames()
        self.ensureModel(True)
        self.model.eval()

    def saveJIT(self):
        self.ensureModel(True)
        self.model.eval()

        with torch.no_grad():
            dummy_image  = torch.randn(1, self.image_channels, self.image_size, self.image_size, dtype=torch.float32, device=self.device)

            #save jit
            #script_file = os.path.join(self.model_folder, 'model_script.pt')
            #torch.jit.save(torch.jit.script(self.model), script_file)
            
            traced_file = os.path.join(self.model_folder, 'model_traced.pt')
        
            traced = torch.jit.trace(self.model, dummy_image)            
            traced.save(traced_file)      


    def classify(self, image) -> tuple[np.ndarray, np.ndarray, int]:
        #if image is opencv image, convert to PIL
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        with torch.no_grad():
            image_tensor : torch.Tensor = self.inference_transforms(image)
            image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(self.device)

            outputs = self.model(image_tensor)
            
            #compute the probabilities
            #_, preds = torch.max(outputs, 1)
            probabilities = nn.functional.softmax(outputs, dim=1)
            probabilities = probabilities[0].cpu().numpy()*100

            outputs = outputs[0].cpu().numpy()

        return outputs, probabilities, outputs.argmax()
    
    def predict_proba_pil(self, image: Image.Image) -> dict[str, float]:
        """
        Predict probabilities for a PIL image.
        Returns dict with class names as keys and probabilities (0-1 scale) as values.
        
        Example:
            {"real": 0.85, "game": 0.15}
        """
        _, probabilities, _ = self.classify(image)
        # Convert from 0-100 scale to 0-1 scale
        probs_dict = {self.class_names[i]: probabilities[i] / 100.0 for i in range(len(self.class_names))}
        return probs_dict
    
    
    