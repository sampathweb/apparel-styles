import time
import copy
import os

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data

import torchvision

try:
    from ml_src.preprocessing import make_dsets, get_label_idx_to_name, image_loader, default_loader, get_transforms
except ImportError:
    from preprocessing import make_dsets, get_label_idx_to_name, image_loader, default_loader, get_transforms


def get_pretrained_model(arch="resnet18", pop_last_pool_layer=False, use_gpu=False):
    if arch == "resnet18":
        resnet = torchvision.models.resnet18(pretrained=True)
        pretrained_features = nn.Sequential(*list(resnet.children())[:-1])
        pretrained_fc = resnet.fc
        fc_dim = 512
    elif arch == "vgg16":
        vgg = torchvision.models.vgg16(pretrained=True)
        pretrained_features = vgg.features
        pretrained_fc = vgg.classifier
        fc_dim = 4096
    elif arch == "alexnet":
        alexnet = torchvision.models.alexnet(pretrained=True)
        pretrained_features = alexnet.features
        pretrained_fc = alexnet.classifier
        fc_dim = 4096
    if pop_last_pool_layer:
        pretrained_features = nn.Sequential(*list(pretrained_features.children())[:-1])
    for param in pretrained_features.parameters():
        param.requires_grad = False
    if use_gpu:
        pretrained_features = pretrained_features.cuda()
    return pretrained_features, pretrained_fc, fc_dim


def optim_scheduler_ft(model, epoch, init_lr=0.01, lr_decay_epoch=7):
    lr = init_lr * (0.1**(epoch//lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print("LR is set to {}".format(lr))

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    return optimizer


class AttributeModel(nn.Module):
    def __init__(self, pretrained_fc, fc_dim, output_dim):
        super().__init__()
        model_steps = list(pretrained_fc.children())[:-1] + [nn.Linear(fc_dim, output_dim)]
        self.model = nn.Sequential(*model_steps)

    def forward(self, x):
        return F.softmax(self.model(x))

class AttributeFCN(nn.Module):
    def __init__(self, in_channels, out_dims, return_conv_layer=False):
        super().__init__()
        self.return_conv_layer = return_conv_layer
        model_steps = [
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, out_dims, 1)
        ]
        model_steps_dummy = [
            # nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_dims, 1)
        ]
        self.conv_model = nn.Sequential(*model_steps)

    def forward(self, x):
        # Get Conv Layer output.  Output channels = number of classes
        classes_conv_out = self.conv_model(x)
        
        # Do Global Average Pooling on the Conv Layer with Number of Channels = Classes
        pool_size = (classes_conv_out.size(2), classes_conv_out.size(3))
        average_pool = F.avg_pool2d(classes_conv_out, kernel_size=classes_conv_out.size()[2:])
        average_pool_flatten = average_pool.view(average_pool.size(0), -1)
        # print(average_pool_flatten)
        classes_softmax = F.log_softmax(average_pool_flatten)
        
        if self.return_conv_layer:
            return classes_conv_out, classes_softmax
        else:
            return classes_softmax
    

def train_attribute_model(model, pretrained_model, train_dset_loader,
                          valid_dset_loader=None,
                          criterion=nn.CrossEntropyLoss(),
                          optim_scheduler=optim_scheduler_ft,
                          use_gpu=None,
                          num_epochs=25, 
                          verbose=False,
                          flatten_pretrained_out=False):
    since = time.time()
    best_model = model
    best_acc = 0.0

    if use_gpu is None:
        use_gpu = torch.cuda.is_available()

    # Define Phases and Get the dataset Sizes
    phases = ["train"]
    dset_sizes = {
        "train": len(train_dset_loader.dataset)
    }
    if valid_dset_loader is not None:
        phases.append("valid")
        dset_sizes["valid"] =  len(valid_dset_loader.dataset)

    if use_gpu:
        pretrained_model.cuda()
        model.cuda()

    for epoch in range(num_epochs):
        if verbose:
            print("Epoch {}/{}".format(epoch, num_epochs - 1))

        # Each epoch has a train and validation Phase
        for phase in phases:
            if phase == "train":
                optimizer = optim_scheduler(model, epoch)
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data
            dset_loader = valid_dset_loader if phase == "valid" else train_dset_loader
            for data in dset_loader:
                # Get the inputs
                batch_files, inputs, labels = data

                # Wrap them in Variable
                if use_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()
                if phase == "train":
                    inputs = Variable(inputs)
                else:
                    inputs = Variable(inputs, volatile=True)
                labels = Variable(labels)
                
                # Get output from pre-trained model and re-shape to Flatten
                out_features = pretrained_model(inputs)
                if flatten_pretrained_out:
                    out_features = out_features.view(out_features.size(0), -1)

                # Forward
                outputs = model(out_features)
                # _, preds = torch.max(outputs.data, 1)
                # loss = criterion(outputs, labels)
                loss = F.nll_loss(outputs, labels)
                preds = outputs.data.max(1)[1]

                # Backward + Optimize only in Training Phase
                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]
            if verbose:
                print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            elif epoch % 3 == 0:
                print("{} Epoch {}/{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch, num_epochs - 1, epoch_loss, epoch_acc))

            # Deep copy the model
            if phase == "valid" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)

    time_elapsed = time.time() - since
    print("Training completed in {:0f}m {:0f}s".format(
        time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {:4f}".format(best_acc))
    return best_model


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def load_model(ModelClass, in_channels, output_dim, weights_path=None, return_conv_layer=False, use_gpu=None):
    model = ModelClass(in_channels, output_dim, return_conv_layer)
    if weights_path:
        model.load_state_dict(torch.load(weights_path))
    if use_gpu:
        model = model.cuda()
    return model

def load_fc_model(pretrained_fc, fc_dim, output_dim, weights_path=None, use_gpu=None):
    model = AttributeModel(pretrained_fc, fc_dim, output_dim)
    if weights_path:
        model.load_state_dict(torch.load(weights_path))
    if use_gpu:
        model = model.cuda()
    return model


def train_model(model, pretrained_features, target_column, labels_file, train_images_folder,
                valid_images_folder=None,
                batch_size=32, num_workers=4, num_epochs=10, 
                use_gpu=None,
                flatten_pretrained_out=False):

    train_dset_loader = make_dsets(train_images_folder, labels_file, target_column,
                                   batch_size=batch_size, num_workers=num_workers, is_train=True)
    valid_dset_loader = None
    if valid_images_folder:
        valid_dset_loader = make_dsets(valid_images_folder, labels_file, target_column,
                                batch_size=batch_size, num_workers=num_workers, is_train=False)

    # Sleeve Length Model
    model = train_attribute_model(model, pretrained_features,
                                train_dset_loader=train_dset_loader,
                                valid_dset_loader=valid_dset_loader,
                                num_epochs=num_epochs, 
                                use_gpu=use_gpu,
                                flatten_pretrained_out=flatten_pretrained_out)
    return model


def save_model(model, weights_path):
    torch.save(model.state_dict(), weights_path)


def predict_model(model, inputs, flatten=False):
    outputs = model(inputs)
    if flatten:
        outputs = outputs.view(outputs.size(0), -1)
    return outputs


import torch.utils.data as data

class AttributePredictDataset(data.Dataset):
    
    def __init__(self, image_url, transform=None, target_transform=None, loader=default_loader):

        super().__init__()
        
        self.image_url = image_url
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        
    def __getitem__(self, index):
        img = self.loader(self.image_url)
        target = 0
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return self.image_url, img, target

    def __len__(self):
        return 1
    
def test_models(attribute_models, pretrained_model, image_url, 
                attribute_idx_map=None, use_gpu=None,
               return_last_conv_layer=False):

    if use_gpu is None:
        use_gpu = torch.cuda.is_available()
    
    image_dset = AttributePredictDataset(image_url, transform=get_transforms(is_train=False))

    dset_loader = data.DataLoader(image_dset, shuffle=False)

    results = {}
    for attrib_name, model in attribute_models.items():
        model.eval()
        for batch_idx, dset in enumerate(dset_loader):
            # Get the inputs
            batch_files, inputs, labels = dset
            # Wrap them in Variable
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            inputs = Variable(inputs)
            labels = Variable(labels)
            # Get output from pre-trained model and re-shape to Flatten
            out_features = pretrained_model(inputs)

            # Forward
            outputs = model(out_features)
            if return_last_conv_layer:
                conv_layer_out = model.conv_model(out_features)
            loss = F.nll_loss(outputs, labels)
            preds_proba, preds = outputs.data.max(1)
            pred_idx = preds.cpu().numpy().flatten()[0]
            pred_proba = np.exp(preds_proba.cpu().numpy().flatten()[0])
            if attribute_idx_map:
                pred_class = attribute_idx_map[attrib_name].get(pred_idx)
                if pred_class:
                    results[attrib_name] = {
                        "pred_idx": pred_idx,
                        "pred_prob": pred_proba,
                        "pred_class": pred_class
                    }
                    if return_last_conv_layer:
                        results[attrib_name]["conv_layer"] = conv_layer_out.data.cpu().numpy()[0, pred_idx]
    return results


def evaluate_model(model, pretrained_model, target_column, labels_file, images_folder,
                batch_size=32, 
                num_workers=4, 
                use_gpu=None,
                flatten_pretrained_out=False):

    if use_gpu is None:
        use_gpu = torch.cuda.is_available()
        
    dset_loader = make_dsets(images_folder, labels_file, target_column,
                            is_train=False, shuffle=False,
                            batch_size=batch_size, 
                            num_workers=num_workers)

    running_loss, running_corrects = (0, 0)
    y_true = []
    y_pred = []
    y_pred_proba = []
    input_files = []
    model.eval()
    for batch_idx, data in enumerate(dset_loader):
        # Get the inputs
        batch_files, inputs, labels = data
        y_true = np.concatenate([y_true, labels.numpy()])
        input_files = np.concatenate([input_files, batch_files])
        # Wrap them in Variable
        if use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        inputs = Variable(inputs)
        labels = Variable(labels)
        # Get output from pre-trained model and re-shape to Flatten
        out_features = pretrained_model(inputs)

        if flatten_pretrained_out:
            out_features = out_features.view(out_features.size(0), -1)

        # Forward
        outputs = model(out_features)
        loss = F.nll_loss(outputs, labels)
        preds_proba, preds = outputs.data.max(1)
        y_pred = np.concatenate([y_pred, preds.cpu().numpy().flatten()])
        y_pred_proba = np.concatenate([y_pred_proba, preds_proba.cpu().numpy().flatten()])

        # Statistics
        running_loss += loss.data[0]
        running_corrects += torch.sum(preds == labels.data)

    return {
        "loss": running_loss,
        "accuracy": running_corrects / len(y_true),
        "input_files": input_files,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba
    }

def predict_attributes(image_url, pretrained_model, attribute_models, attribute_idx_map=None, 
                       flatten_pretrained_out=True, use_gpu=None):

    if use_gpu is None:
        use_gpu = torch.cuda.is_available()
    
    image_features = image_loader(image_url, use_gpu=use_gpu)

    print(image_features.size())
    pretrained_features = predict_model(pretrained_model, image_features, flatten=flatten_pretrained_out)
    results = {}
    for attrib_name, model in attribute_models.items():
        print("Predicting {}".format(attrib_name))        
        outputs = predict_model(model, pretrained_features)
        pred_prob, pred_class = outputs.data.max(1)        
        if use_gpu:
            pred_prob = pred_prob.cpu()
            pred_class = pred_class.cpu()
        pred_prob = np.exp(pred_prob.numpy().flatten()[0])
        pred_class = pred_class.numpy().flatten()[0]
        # outputs_arr = outputs_arr.numpy()
        # pred_prob, pred_class = outputs_arr.max(), outputs_arr.argmax()
        if attribute_idx_map:
            pred_class = attribute_idx_map[attrib_name].get(pred_class)
        if pred_class is not None:
            results[attrib_name] = (pred_class, pred_prob)
    return results

def create_attributes_fc_model(pretrained_fc, pretrained_features, fc_dim, target_columns, weights_root,
                            labels_file, train_images_folder, valid_images_folder=None, is_train=True,
                            batch_size=32, num_workers=4, num_epochs=10,  use_gpu=None):
    models = {}
    for col_name, col_dim in target_columns.items():
        print("Processing Attribute: {}".format(col_name))
        weights_path = os.path.join(weights_root, col_name + ".pth")
        load_weights_path = None
        if os.path.exists(weights_path):
            load_weights_path = weights_path
        model = load_fc_model(pretrained_fc, fc_dim, col_dim, weights_path=load_weights_path, use_gpu=use_gpu)
        if is_train:
            print("Start Training for: {}".format(col_name))
            model = train_model(model, pretrained_features, col_name, labels_file, train_images_folder,
                                valid_images_folder,
                                batch_size, num_workers, num_epochs,
                                use_gpu=use_gpu,
                               flatten_pretrained_out=True)
        save_model(model, weights_path)
        models[col_name] = model
    return models


def create_attributes_model(ModelClass, in_dims, pretrained_features, target_columns, weights_root,
                            labels_file, train_images_folder, valid_images_folder=None, is_train=True,
                            batch_size=32, num_workers=4, num_epochs=10, use_gpu=None):
    models = {}
    for col_name, col_dim in target_columns.items():
        print("Processing Attribute: {}".format(col_name))
        weights_path = os.path.join(weights_root, col_name + ".pth")
        load_weights_path = None
        if os.path.exists(weights_path):
            load_weights_path = weights_path
        model = load_model(ModelClass, in_dims, col_dim, weights_path=load_weights_path, use_gpu=use_gpu)
        if is_train:
            print("Start Training for: {}".format(col_name))
            model = train_model(model, pretrained_features, col_name, labels_file, train_images_folder, valid_images_folder,
                               batch_size, num_workers, num_epochs, use_gpu=use_gpu)
        save_model(model, weights_path)
        models[col_name] = model
    return models


def visualize_model(model, dset_loader, num_images=5, use_gpu=None):
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()

    for i, data in enumerate(dset_loader):
        batch_files, inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        plt.figure()

        imshow(inputs.cpu().data[0])
        plt.title('pred: {}'.format(dset_classes[labels.data[0]]))
        plt.show()

        if i == num_images - 1:
            break