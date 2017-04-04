import torch.utils.data as data

import os.path
from PIL import Image
from glob import glob
from shutil import copyfile, rmtree

import numpy as np
import scipy.io
import pandas as pd

import torch
from torch.autograd import Variable
from torchvision import transforms


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def train_valid_test_split(SOURCE_DATA_DIR, TARGET_DATA_DIR, train_size=0.8, valid_size=0.1, 
                           COMBINE_FOLDERS=None, SELECT_FOLDERS=None):
    """
    Usage:
        SOURCE_DATA_DIR = "data/ClothingAttributeDataset/images/"
        TARGET_DATA_DIR = "data/"

        train_valid_test_split(SOURCE_DATA_DIR, TARGET_DATA_DIR)
    """
    if COMBINE_FOLDERS is None:
        COMBINE_FOLDERS = dict()
    for folder_name in ["train", "test", "valid"]:
        rmtree(os.path.join(TARGET_DATA_DIR, folder_name), ignore_errors=True)
        os.makedirs(os.path.join(TARGET_DATA_DIR, folder_name))

    # Split records by 80-20 between Train and Validation Set
    filenames = np.random.permutation(glob(os.path.join(SOURCE_DATA_DIR, "*.jpg")))

    train_idx = int(len(filenames) * train_size)
    test_idx = int(len(filenames) * (train_size+valid_size))
    for idx, filename in enumerate(filenames):
        target_name = filename.split("/")[-1]
        if idx < train_idx:
            target_filepath = os.path.join(TARGET_DATA_DIR, "train", target_name)
        elif idx < test_idx:
            target_filepath = os.path.join(TARGET_DATA_DIR, "valid", target_name)
        else:
            target_filepath = os.path.join(TARGET_DATA_DIR, "test", target_name)            
        copyfile(filenames[idx], target_filepath)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')

def get_label_values_to_idx(label_values, label_name):
    return label_values["values_to_idx"][label_name]

def load_label_values(label_values_file):
    import json

    with open(label_values_file, 'r') as f:
        label_values = json.load(f)
    
    # Covert String numbers to integers
    for key, values in label_values["idx_to_names"].items():
        label_values["idx_to_names"][key] = {int(k): v for k, v in values.items()}

    for key, values in label_values["values_to_idx"].items():
        label_values["values_to_idx"][key] = {k: int(v) for k, v in values.items()}
        
    return label_values

def get_attribute_dims(label_values_file):
    label_values = load_label_values(label_values_file)
    return label_values["attribute_dims"]

def get_label_idx_to_name(label_values, attribute_name, label_idx):
    return label_values["idx_to_names"][attribute_name][label_dx]

def combine_labels(LABEL_DIR, labels_file, label_values_file, na_value=None, combine_pattern_cols=True):
    dfs = []
    for label_file in glob(os.path.join(LABEL_DIR, "*.mat")):
        feature_name = label_file.split("/")[-1].split(".")[0]
        values = (scipy.io.loadmat(label_file)["GT"] - 1)
        feat_df = pd.DataFrame(data=values, index=np.arange(1, len(values)+1), columns=[feature_name])
        if na_value:
            feat_df.fillna(na_value, inplace=True)
        dfs.append(feat_df)
    labels_df = pd.concat(dfs, axis=1)
    labels_df["image"] = labels_df.index.map(lambda val: "{:06d}.jpg".format(val))
    
    # Combine Pattern Columns
    label_values = load_label_values(label_values_file)
    if combine_pattern_cols:
        pattern_cols = [col for col in labels_df.columns if "pattern_" in col]
        labels_df["pattern_GT"] = labels_df[pattern_cols].apply(lambda row: row.argmax(), axis=1)
        labels_df["pattern_GT"] = labels_df["pattern_GT"].map(
            get_label_values_to_idx(label_values, "pattern_GT"), na_action="ignore")
        
    labels_df.to_csv(labels_file, index=False)
    
    labels_df.set_index("image", inplace=True)
    return labels_df

def get_labels(labels_file):
    labels_df = pd.read_csv(labels_file)
    labels_df.set_index("image", inplace=True)
    return labels_df    


class AttributeDataset(data.Dataset):
    
    def __init__(self, images_folder, labels_df, target_column, transform=None, target_transform=None,
                 loader=default_loader):

        super().__init__()
        
        self.images_folder = images_folder
        # Index should be the filename in the root folder
        self.labels_df = labels_df
        self.target_column = target_column
        # self.class_to_idx = { target_col: idx for target_col in self.target_columns }
        
        self.imgs = self._get_data()
        
        if len(self.imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + images_folder + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        
    def _get_data(self):
        images = []
        
        for file_location in glob(os.path.join(self.images_folder, "*.jpg")):
            filename = file_location.split("/")[-1]
            target_value = self.labels_df.loc[filename, self.target_column]
            if not np.isnan(target_value):
                item = (file_location, int(target_value))
                images.append(item)
        return images

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return path, img, target

    def __len__(self):
        return len(self.imgs)


def get_transforms(is_train=False):
    if is_train:
        data_transforms = transforms.Compose([
            transforms.Scale(266),
            transforms.CenterCrop((400, 266)),
            # transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        data_transforms = transforms.Compose([
            transforms.Scale(266),
            transforms.CenterCrop((400, 266)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return data_transforms


def make_dsets(IMAGES_FOLDER, LABELS_FILE, target_column, batch_size=32, num_workers=4, 
               is_train=True, shuffle=True):
    
    # Data Augmentation and Normalization
    data_transforms = get_transforms(is_train)

    labels_df = get_labels(LABELS_FILE)
    dset = AttributeDataset(IMAGES_FOLDER, labels_df, target_column=target_column,
                                           transform=data_transforms)

    dset_loader = data.DataLoader(dset, batch_size=batch_size,
                                            shuffle=shuffle, num_workers=num_workers)
    return dset_loader


def image_loader(image_name, transforms=None, use_gpu=None, requires_grad=False):
    """load image, returns cuda tensor"""
    if transforms is None:
        transforms = get_transforms(is_train=False)
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()
        
    # image = Image.open(image_name)
    image = default_loader(image_name)
    image = transforms(image)
    image = Variable(image, requires_grad=requires_grad)
    image = image.unsqueeze(0)  # Add a top level dimension for Batches
    if use_gpu:
        image = image.cuda()
    return image