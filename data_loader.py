import numpy as np
from PIL import Image
import torch.utils.data as data


class SYSUData(data.Dataset):
    def __init__(self, data_dir,  transform=None, colorIndex = None, thermalIndex = None , model_test = False):
        if not model_test:
            # Load training images (path) and labels
            train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
            self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

            train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
            self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')
        else :
            print("model_testing")
            train_color_image = np.load(data_dir + 'subset_train_rgb_resized_img.npy')
            self.train_color_label = np.load(data_dir + 'subset_train_rgb_resized_label.npy')

            train_thermal_image = np.load(data_dir + 'subset_train_ir_resized_img.npy')
            self.train_thermal_label = np.load(data_dir + 'subset_train_ir_resized_label.npy')
        
        # BGR to RGB
        self.train_color_image   = train_color_image
        self.train_thermal_image = train_thermal_image
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)
        
        
class RegDBData(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex = None, thermalIndex = None):
        # Load training images (path) and labels
        train_color_list   = data_dir + 'idx/train_visible_{}'.format(trial)+ '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial)+ '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)
        
        train_color_image = []
        for i in range(len(color_img_file)):
   
            img = Image.open(data_dir+ color_img_file[i])
            # 修改成和版本没有关系的 ， img = img.resize((192, 384), Image.Resampling.LANCZOS)

            img = img.resize((192, 384), Image.Resampling.LANCZOS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image) 
        
        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir+ thermal_img_file[i])
            img = img.resize((192, 384), Image.Resampling.LANCZOS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)
        
        # BGR to RGB
        self.train_color_image = train_color_image  
        self.train_color_label = train_color_label
        
        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label
        
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)
        
class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (192,384)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.Resampling.LANCZOS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)
        
class TestDataOld(data.Dataset):
    def __init__(self, data_dir, test_img_file, test_label, transform=None, img_size = (192,384)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(data_dir + test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.Resampling.LANCZOS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)        
def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label

if __name__ == "__main__":
    data_dir = "./Datasets/SYSU-MM01/"
    # 加载 RGB 图像和标签数据
    train_rgb_resized_img = np.load(data_dir + 'train_rgb_resized_img.npy')
    print(train_rgb_resized_img[0].shape)
    # train_rgb_resized_label = np.load(data_dir + 'train_rgb_resized_label.npy')

    # # 加载 IR 图像和标签数据
    # train_ir_resized_img = np.load(data_dir + 'train_ir_resized_img.npy')
    # train_ir_resized_label = np.load(data_dir + 'train_ir_resized_label.npy')

    # # 从每个数据集中按顺序取20个数据点
    # rgb_images_subset = train_rgb_resized_img[:20]
    # rgb_labels_subset = train_rgb_resized_label[:20]
    # ir_images_subset = train_ir_resized_img[:20]
    # ir_labels_subset = train_ir_resized_label[:20]

    # # 保存到新的文件中
    # np.save(data_dir + 'subset_train_rgb_resized_img.npy', rgb_images_subset)
    # np.save(data_dir + 'subset_train_rgb_resized_label.npy', rgb_labels_subset)
    # np.save(data_dir + 'subset_train_ir_resized_img.npy', ir_images_subset)
    # np.save(data_dir + 'subset_train_ir_resized_label.npy', ir_labels_subset)