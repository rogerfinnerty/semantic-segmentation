"""
CS585 HW4 Semantic Segmentation
Roger Finnerty, Demetrios Kechris, Benjamin Burnham
April 1, 2024

"""

import os
import torch
import fcn_model3
import fcn_dataset
import numpy as np
from PIL import Image
from tqdm import tqdm

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model
num_classes = 32
model = fcn_model3.FCN8s(num_classes).to(device)

# Define the dataset and dataloader
images_dir_train = "train/"
labels_dir_train = "train_labels/"
class_dict_path = "class_dict.csv"
resolution = (384, 512)
batch_size = 16
num_epochs = 50

camvid_dataset_train = fcn_dataset.CamVidDataset(root='CamVid/', images_dir=images_dir_train, labels_dir=labels_dir_train, class_dict_path=class_dict_path, resolution=resolution, crop=True)
dataloader_train = torch.utils.data.DataLoader(camvid_dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)

images_dir_val = "val/"
labels_dir_val = "val_labels/"
camvid_dataset_val = fcn_dataset.CamVidDataset(root='CamVid/', images_dir=images_dir_val, labels_dir=labels_dir_val, class_dict_path=class_dict_path, resolution=resolution, crop=False)
dataloader_val = torch.utils.data.DataLoader(camvid_dataset_val, batch_size=1, shuffle=False, num_workers=4, drop_last=False)

images_dir_test = "test/"
labels_dir_test = "test_labels/"
camvid_dataset_test = fcn_dataset.CamVidDataset(root='CamVid/', images_dir=images_dir_test, labels_dir=labels_dir_test, class_dict_path=class_dict_path, resolution=resolution, crop=False)
dataloader_test = torch.utils.data.DataLoader(camvid_dataset_test, batch_size=1, shuffle=False, num_workers=4, drop_last=False)

# Define the loss function and optimizer
def loss_fn(outputs, labels):
    """
    Computes the cross entropy loss between the model outputs and 
    labeled data
    """
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    # raise NotImplementedError("Implement the loss function")
    return loss


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def pixel_accuracy(labels, predictions):
    correct_pixels = torch.sum(labels == predictions).item()
    total_pixels = labels.numel()
    return correct_pixels / total_pixels

def mean_IOU(labels, predictions):
    iou_sum = 0.0
    for i in range(num_classes):
        intersection = torch.sum((labels == i) & (predictions == i)).item()
        union = torch.sum((labels == i) | (predictions == i)).item()
        if union == 0:
            iou_sum += 1.0  
        else:
            iou_sum += intersection / union
    return iou_sum / num_classes

def frequency_IOU(labels, predictions):
    class_counts = torch.zeros(num_classes, device=device)
    for i in range(num_classes):
        class_counts[i] = torch.sum(labels == i)
    
    freq_weights = class_counts / torch.sum(class_counts)
    
    weighted_iou = 0.0
    for i in range(num_classes):
        intersection = torch.sum((labels == i) & (predictions == i)).item()
        union = torch.sum((labels == i) | (predictions == i)).item()
        if union == 0:
            weighted_iou += freq_weights[i].item() * 1.0 
        else:
            weighted_iou += freq_weights[i].item() * (intersection / union)
    return weighted_iou

def eval_model(model, dataloader, device, save_pred=False):
    model.eval()
    loss_list = []
    if save_pred:
        pred_list = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss_list.append(loss.item())
            _, predicted = torch.max(outputs, 1)

            if save_pred:
                pred_list.append(predicted.cpu().numpy())

        pixel_acc = pixel_accuracy(labels, predicted)
        mean_iou = mean_IOU(labels, predicted)
        freq_iou = frequency_IOU(labels, predicted)

        loss = sum(loss_list) / len(loss_list)
        print('Pixel accuracy: {:.4f}, Mean IoU: {:.4f}, Frequency weighted IoU: {:.4f}, Loss: {:.4f}'.format(pixel_acc, mean_iou, freq_iou, loss))

    if save_pred:
        pred_list = np.concatenate(pred_list, axis=0)
        np.save('test_pred.npy', pred_list)
    model.train()

def visualize_model(model, dataloader, device):
    log_dir = "vis/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    cls_dict = dataloader.dataset.class_dict.copy()
    cls_list = [cls_dict[i] for i in range(len(cls_dict))]
    model.eval()
    with torch.no_grad():
        for ind, (images, labels) in enumerate(tqdm(dataloader)):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            images_vis = fcn_dataset.rev_normalize(images)
            # Save the images and labels
            img = images_vis[0].permute(1, 2, 0).cpu().numpy()
            img = img * 255
            img = img.astype('uint8')
            label = labels[0].cpu().numpy()
            pred = predicted[0].cpu().numpy()

            label_img = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
            pred_img = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
            for j in range(len(cls_list)):
                mask = label == j
                label_img[mask] = cls_list[j][0]
                mask = pred == j
                pred_img[mask] = cls_list[j][0]
            # horizontally concatenate the image, label, and prediction, and save the visualization
            vis_img = np.concatenate([img, label_img, pred_img], axis=1)
            vis_img = Image.fromarray(vis_img)
            vis_img.save(os.path.join(log_dir, 'img_{:04d}.png'.format(ind)))
            
    model.train()
    
if __name__ == '__main__':
    # Train the model
    print('training model')
    loss_list = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(dataloader_train):
            images, labels = images.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

            if (i+1) % 10 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(dataloader_train), sum(loss_list)/len(loss_list)))
                loss_list = []

        # eval the model        
        print('Post epoch evaluation begin')
        eval_model(model, dataloader_val, device)
        print('Post epoch evaluation complete')

    print('='*20)
    print('Finished Training, evaluating the model on the test set')
    eval_model(model, dataloader_test, device, save_pred=True)

    print('='*20)
    print('Visualizing the model on the test set, the results will be saved in the vis/ directory')
    visualize_model(model, dataloader_test, device)