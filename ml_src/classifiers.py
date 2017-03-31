import time
import copy
import os

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision

from .preprocessing import make_dsets, get_label_idx_to_name, image_loader


def get_pretrained_model(arch="resnet18", top_layer=False):
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
    for param in pretrained_features.parameters():
        param.requires_grad = False
    return pretrained_features, pretrained_fc, fc_dim


def optim_scheduler_ft(model, epoch, init_lr=0.001, lr_decay_epoch=7):
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


def train_attribute_model(model, pretrained_model, train_dset_loader,
                          valid_dset_loader=None,
                          criterion=nn.CrossEntropyLoss(),
                          optim_scheduler=optim_scheduler_ft,
                          use_gpu=None,
                          num_epochs=25):
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
        print("Epoch {}/{}".format(epoch, num_epochs - 1))

        # Each epoch has a train and validation Phase
        for phase in phases:
            if phase == "train":
                optimizer = optim_scheduler(model, epoch)

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data
            dset_loader = valid_dset_loader if phase == "valid" else train_dset_loader
            for data in dset_loader:
                # Get the inputs
                inputs, labels = data

                # Wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                                             Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Get output from pre-trained model and re-shape to Flatten
                out_features = pretrained_model(inputs)
                out_features = out_features.view(out_features.size(0), -1)

                # Forward
                print_out
                outputs = model(out_features)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # Backward + Optimize only in Training Phase
                if phase == "train":
                    loss.backward()
                    optimizer.step()

                # Statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

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


def load_model(pretrained_fc, fc_dim, output_dim, weights_path=None):
    model = AttributeModel(pretrained_fc, fc_dim, output_dim)
    if weights_path:
        model.load_state_dict(torch.load(weights_path))
    return model


def train_model(model, pretrained_features, target_column, labels_file, train_images_folder, valid_images_folder=None,
                     batch_size=32, num_workers=4, num_epochs=10):

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
                                num_epochs=num_epochs)
    return model


def save_model(model, weights_path):
    torch.save(model.state_dict(), weights_path)


def predict_model(model, inputs, flatten=False):
    outputs = model(inputs)
    if flatten:
        outputs = outputs.view(outputs.size(0), -1)
    return outputs


def predict_attributes(image_url, pretrained_model, attribute_models, attribute_idx_map=None):
    image_features = image_loader(image_url)
    pretrained_features = predict_model(pretrained_model, image_features, flatten=True)
    results = {}
    for attrib_name, model in attribute_models.items():
        outputs = predict_model(model, pretrained_features)
        outputs_arr = outputs.data.numpy()
        pred_prob, pred_class = outputs_arr.max(), outputs_arr.argmax()
        if attribute_idx_map:
            pred_class = attribute_idx_map[attrib_name].get(pred_class)
        if pred_class is not None:
            results[attrib_name] = (pred_class, pred_prob)
    return results

def create_attributes_model(pretrained_fc, pretrained_features, fc_dim, target_columns, weights_root,
                            labels_file, train_images_folder, valid_images_folder=None, is_train=True,
                            batch_size=32, num_workers=4, num_epochs=10):
    models = {}
    for col_name, col_dim in target_columns.items():
        print("Processing Attribute: {}".format(col_name))
        weights_path = os.path.join(weights_root, col_name + ".pth")
        load_weights_path = None
        if os.path.exists(weights_path):
            load_weights_path = weights_path
        model = load_model(pretrained_fc, fc_dim, col_dim, weights_path=load_weights_path)
        if is_train:
            print("Start Training for: {}".format(col_name))
            model = train_model(model, pretrained_features, col_name, labels_file, train_images_folder, valid_images_folder,
                               batch_size, num_workers, num_epochs)
        save_model(model, weights_path)
        models[col_name] = model
    return models


def visualize_model(model, dset_loader, num_images=5, use_gpu=None):
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()

    for i, data in enumerate(dset_loader):
        inputs, labels = data
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