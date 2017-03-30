class ClothingAttributesDataset(data.Dataset):
    
    def __init__(self, images_folder, labels_df, target_columns=None, transform=None, target_transform=None,
                 loader=default_loader):

        super().__init__()
        
        self.images_folder = images_folder
        # Index should be the filename in the root folder
        self.labels_df = labels_df
        self.target_columns = target_columns
        if self.target_columns is None:
            self.target_columns = list(self.labels_df.columns)
        # self.class_to_idx = { target_col: idx for target_col in self.target_columns }
        
        self.imgs = self.make_dataset()
        
        if len(self.imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        
    def make_dataset(self):
        images = []
        
        for file_location in glob(os.path.join(self.images_folder, "*.jpg")):
            filename = file_location.split("/")[-1]
            y = [self.labels_df.loc[filename, target_col] for target_col in self.target_columns]
            item = (file_location, y)
            images.append(item)

        return images

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, np.array(target)

    def __len__(self):
        return len(self.imgs)

    
self.model = nn.Sequential(*[
            nn.Linear(in_features, 4096),
            nn.ReLU(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, output_dims)
        ])


nn.Sequential(*list(vgg.classifier.children())[:-1])
import torch

torch.save(model.state_dict(), 'weights/example.pth')
# net.load_state_dict(torch.load('./net.pth'))
model.load_state_dict(torch.load('weights/example.pth'))
model_1.load_state_dict(torch.load('weights/example.pth'))