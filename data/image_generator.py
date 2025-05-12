import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import Sequence

class ImageLabelGenerator(Sequence):
    def __init__(self, data_dir, batch_size=32, input_size=(256, 256), shuffle=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        
        self.A_dir = os.path.join(data_dir, 'A')
        self.B_dir = os.path.join(data_dir, 'B')
        self.label_dir = os.path.join(data_dir, 'label')
        
        # Get list of all image names (assuming files in A, B, and label are in correct order)
        self.image_names = sorted(os.listdir(self.A_dir))
        
        # Shuffle image names if required
        if self.shuffle:
            np.random.shuffle(self.image_names)

    def __len__(self):
        return int(np.floor(len(self.image_names) / self.batch_size))

    def __getitem__(self, index):
        batch_names = self.image_names[index * self.batch_size: (index + 1) * self.batch_size]
        
        images_A = []
        images_B = []
        labels = []
        
        for name in batch_names:
            img_A = self.load_and_preprocess_image(os.path.join(self.A_dir, name))
            images_A.append(img_A)
            
            img_B = self.load_and_preprocess_image(os.path.join(self.B_dir, name))
            images_B.append(img_B)
            
            label_img = self.load_and_preprocess_image(os.path.join(self.label_dir, name), label=True)
            labels.append(label_img)
        
        images_A = np.array(images_A)
        images_B = np.array(images_B)
        labels = np.array(labels)
        
        inputs = np.concatenate([images_A, images_B], axis=-1)
        
        return inputs, labels

    def load_and_preprocess_image(self, img_path, label=False):
        
        #img = image.load_img(img_path, target_size=self.input_size)
        img = image.load_img(img_path, target_size=self.input_size, color_mode='rgb' if not label else 'grayscale')
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        
        if label:
            img_array = np.expand_dims(img_array, axis=-1)
        
        return img_array

    def on_epoch_end(self):
        # Shuffle the dataset at the end of each epoch if required
        if self.shuffle:
            np.random.shuffle(self.image_names)
