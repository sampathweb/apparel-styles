from preprocessing import train_valid_test_split, combine_labels, get_attribute_dims
from classifiers import get_pretrained_model, create_attributes_model, AttributeFCN
from utils import is_gpu_available

# Train-Test Split Folders
SOURCE_DATA_DIR = "data/ClothingAttributeDataset/images/"
TARGET_DATA_DIR = "data/ClothingAttributeDataset/"

# Labels File
LABEL_DIR = "data/ClothingAttributeDataset/labels/"
labels_file = "data/labels.csv"
label_values_file = "data/label_values.json"

# Train and Validation Images
TRAIN_IMAGES_FOLDER = "data/ClothingAttributeDataset/train/"
VALID_IMAGES_FOLDER = "data/ClothingAttributeDataset/valid/"
TEST_IMAGES_FOLDER = "data/ClothingAttributeDataset/test/"


if __name__ == "__main__":

    target_dims = get_attribute_dims(label_values_file)
    use_gpu = is_gpu_available()
    pretrained_conv_model, _, _ = get_pretrained_model("vgg16", pop_last_pool_layer=True, use_gpu=use_gpu)


    attribute_models = create_attributes_model(AttributeFCN, 512, pretrained_conv_model,
                                    target_dims, 
    #                                 dict(list(target_dims.items())[:3]),
                                    "weights/vgg16-fcn-266-2/",
                                    labels_file, 
                                     TRAIN_IMAGES_FOLDER, 
                                     VALID_IMAGES_FOLDER, 
                                     num_epochs=50,
                                     is_train=True,
                                     use_gpu=use_gpu)
