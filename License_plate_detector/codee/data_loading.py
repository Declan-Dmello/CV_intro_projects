import os
import numpy as np



ds_path = "../data/archive"

def get_data_info(ds_path):
    train_images_path = os.path.join(ds_path, "images", "train")
    val_images_path = os.path.join(ds_path, "images", "val")
    train_label_path = os.path.join(ds_path, "labels", "train")
    val_label_path = os.path.join(ds_path, "labels", "val")

    train_images =   os.listdir(train_images_path) if os.path.exists(train_images_path) else []
    val_images = os.listdir(val_images_path) if os.path.exists(val_images_path) else []
    train_labels = os.listdir(train_label_path) if os.path.exists(train_label_path) else []
    val_labels = os.listdir(val_label_path) if os.path.exists(val_label_path) else []

    print(f"The train images are :{len(train_images)} ")
    print(f"The val images  are  :{len(val_images)} ")
    print(f"The train labels  are:{len(train_labels)} ")
    print(f"The val labels       :{len(val_labels)} ")

    #checking images and label matching
    image_files = {os.path.splitext(f)[0] for f in os.listdir(train_images_path) if f.endswith((".jpg" , ".jpeg" , ".png"))}
    label_file = {os.path.splitext(f)[0] for f in os.listdir(train_label_path)  if f.endswith(".txt")}

    images_wo_labels = image_files - label_file
    labels_wo_images =  label_file -  image_files

    print("DI")
    print(images_wo_labels)
    print("Yoo")
    print(labels_wo_images)


get_data_info(ds_path)




