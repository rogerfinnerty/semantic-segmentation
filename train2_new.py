"""
CS585 HW4 Semantic Segmentation
Roger Finnerty, Demetrios Kechris, Benjamin Burnham
April 1, 2024

"""

"""
NOTE:
The structure of this code was altered to avoid redundant calculations
and to reduce solve time. Besides structure, the supplied code was
untouched.

"""

import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import fcn_model_new
import fcn_dataset_new

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

######### Metric Functions #########
# 1 - Pixel Accuracy
def calc_pixel_acc(confusion_mat):
    correct = confusion_mat.diag().sum().item()
    total = confusion_mat.sum().item()
    return correct / total

# IOU Calc
def calc_iou(confusion_mat):
    intersection = confusion_mat.diag()
    union = confusion_mat.sum(dim=0) + confusion_mat.sum(dim=1) - intersection
    return intersection, union

# 2 - Mean IOU
def calc_mean_iou(intersection, union):
    iou = intersection / union
    return torch.nanmean(iou).item()

# 3 - Weighted Frequency IOU
def calc_freq_iou(confusion_mat, intersection, union, epsilon = 1e-8):
    iou = intersection / (union + epsilon)
    frequency = confusion_mat.sum(dim=1) / confusion_mat.sum()
    return (frequency * iou).sum().item()


def eval_model(model, dataloader, device, save_pred=False):
    model.eval()
    loss_list = []

    # Initialize confusion matrix with zeros
    confusion_mat = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    if save_pred:
        pred_list = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss_list.append(loss.item())
            _, predicted = torch.max(outputs, 1)

            # Populate confusion matrix for metric calcs
            indices = (labels.view(-1), predicted.view(-1))
            vals = torch.ones_like(labels.view(-1), device=device)
            confusion_mat.index_put_(indices, vals, accumulate=True)

            if save_pred:
                pred_list.append(predicted.cpu().numpy())
        
        # Metric calcs
        pixel_acc = calc_pixel_acc(confusion_mat)
        intersection, union = calc_iou(confusion_mat)
        mean_iou =calc_mean_iou(intersection, union)
        freq_iou = calc_freq_iou(confusion_mat, intersection, union)

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
            
            images_vis = fcn_dataset_new.rev_normalize(images)
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
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the model
    num_classes = 32
    model = fcn_model_new.FCN8s(num_classes).to(device)

    # Define the dataset and dataloader
    images_dir_train = "train/"
    labels_dir_train = "train_labels/"
    class_dict_path = "class_dict.csv"
    resolution = (384, 512)
    batch_size = 16
    num_epochs = 50

    camvid_dataset_train = fcn_dataset_new.CamVidDataset(root='CamVid/', images_dir=images_dir_train, labels_dir=labels_dir_train, class_dict_path=class_dict_path, resolution=resolution, crop=True)
    dataloader_train = torch.utils.data.DataLoader(camvid_dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    print('Training set ready')

    images_dir_val = "val/"
    labels_dir_val = "val_labels/"
    camvid_dataset_val = fcn_dataset_new.CamVidDataset(root='CamVid/', images_dir=images_dir_val, labels_dir=labels_dir_val, class_dict_path=class_dict_path, resolution=resolution, crop=False)
    dataloader_val = torch.utils.data.DataLoader(camvid_dataset_val, batch_size=1, shuffle=False, num_workers=4, drop_last=False)
    print('Validation set ready')

    images_dir_test = "test/"
    labels_dir_test = "test_labels/"
    camvid_dataset_test = fcn_dataset_new.CamVidDataset(root='CamVid/', images_dir=images_dir_test, labels_dir=labels_dir_test, class_dict_path=class_dict_path, resolution=resolution, crop=False)
    dataloader_test = torch.utils.data.DataLoader(camvid_dataset_test, batch_size=1, shuffle=False, num_workers=4, drop_last=False)
    print('Test set ready')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    loss_list = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(dataloader_train):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Backward pass and optimize
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