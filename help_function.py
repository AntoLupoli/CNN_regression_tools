from typing import List, Tuple
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def print_train_time(start: float, end: float, device: torch.device = None):

    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: List[str],
                        image_size: Tuple[int, int] = (224, 224),
                        transform: torchvision.transforms = None,
                        device: torch.device=device):


    # 2. Open image
    img = Image.open(image_path)

    # 3. Create transformation for image (if it doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        # Change manually the value below
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    ### Predict on image ###

    # 4. Make sure the model is on the target device
    model.to(device)

    model.eval()
    with torch.inference_mode():
      # 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
      transformed_image = image_transform(img).unsqueeze(dim=0)

      # 7. Make a prediction on image with an extra dimension and send it to the target device
      target_image_pred = model(transformed_image.to(device))

    # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 9. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 10. Plot image with predicted label and probability
    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
    plt.axis(False)

import random
from pathlib import Path

def multi_plot(
        num_images_to_plot: int,
        model: torch.nn.Module,
        directory_images: str,
        class_names: List[str],
        transforms: torchvision.transforms,
):

    test_dir = directory_images
    test_image_path_list = list(Path(test_dir).glob("*/*.jpg")) # get list all image paths from test data
    test_image_path_sample = random.sample(population=test_image_path_list, # go through all of the test image paths
                                           k=num_images_to_plot) # randomly select 'k' image paths to pred and plot

    # Make predictions on and plot the images
    for image_path in test_image_path_sample:
        pred_and_plot_image(model=model,
                            image_path=image_path,
                            class_names=class_names,
                            # transform=weights.transforms(), # optionally pass in a specified transform from our pretrained model weights
                            image_size=(224, 224),
                            transform = transforms)
from matplotlib_venn import venn3, venn3_circles

def check_dataleakage(train_idx_folds: list, val_idx_folds: list, test_idx_folds:list, show_graph = False):
    '''

    Args:
        number_fold: number of splits
        train_idx_folds: [ [train_idx_fold1], [train_idx_fold2],..., [train_idx_fold_n] ]
        val_idx_folds: As above
        test_idx_folds: As above
        show_graph: show the graph

    Returns:
        If there are the same index in the each set (data leakege), you can also print the intersection graph
    '''

    number_fold = len(train_idx_folds)
    # Numerical check
    cnt=0
    for fold in range(number_fold):
        train_set = set(train_idx_folds[fold])
        val_set = set(val_idx_folds[fold])
        test_set = set(test_idx_folds[fold])


        if len(train_set & val_set) or len( train_set & test_set) or len(val_set & test_set) > 0:
            print(f"DATA LEAKAGE FOUNDED INSIDE FOLD{fold+1}")
            cnt=+1
            if len(train_set & val_set):
                print(f'train_set vs val_set: {train_set & val_set}')
            if len( train_set & test_set):
                print( f'train_set & test_set: {train_set & test_set}')
            if len(val_set & test_set):
                print(f'val_set & test_set: {val_set & test_set}')
    if cnt == 0:
         print(f"NO DATA LEAKAGE FOUNDED")


    # Graphic Check
    if show_graph:
        plt.figure(figsize=(8, 8))
        plt.title("Check Data Leakage (Overlap)")

        # venn3 accept 3 list, venn2 accept 2 list and so on
        venn3([train_set, val_set, test_set], set_labels=('Train Set', 'Validation Set', 'Test set'))
        venn3_circles(subsets=[train_set, val_set, test_set], linestyle="dashed", linewidth=2)

        if len(train_set & val_set) or len( train_set & test_set) or len(val_set & test_set) > 0:
            plt.text(0.2, -0.7, "DATA LEAKAGE FOUNDED",
                     ha='center', color='red', fontsize=12, weight='bold')
        else:
            plt.text(0.2, -0.7, "IT'S OK, NO OVERLAP",
                     ha='center', color='green', fontsize=12, weight='bold')
        plt.show()


def get_misclassified_info(model, dataset, test_indices, device, transform):
    """
    Returns a dictionary list with wrong images paths
    """
    #image_to_tensor = transforms.ToTensor()
    misclassified_list = []
    model.eval()

    with torch.inference_mode():
        for idx in test_indices:
            img, true_label = dataset[idx]

            img_input = transform(img).unsqueeze(0).to(device)

            image_path = dataset.samples[idx][0]

            logits = model(img_input)
            pred_label = torch.argmax(logits, dim=1).item()

            # Save path and details if there's an error
            if pred_label != true_label:
                misclassified_list.append({
                    "path": image_path,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "true_name": dataset.classes[true_label],
                    "pred_name": dataset.classes[pred_label]
                })

    return misclassified_list

def save_misclassified_report( errors, dir_path: str , fold: int):
    import os
    import csv
    '''

    Args:
        errors: the misclassified list
        dir_path: directory to save the report
        fold: number of folds

    Returns:

    '''


    '''
    error_file_path = f'{dir_path}/errors_fold_{fold + 1}.txt'
    print(f"Saving errors in: {error_file_path} directory")

    with open(error_file_path, 'w') as f:
        f.write(f"=== REPORT ERRORS FOLD {fold + 1} ===\n")
        f.write(f"TOT ERRORS: {len(errors)}\n")
        f.write("-" * 50 + "\n")

        for err in errors:
            f.write(f"Path: {err['path']}\n")
            f.write(
            f"      True: {err['true_name']} ({err['true_label']}) -> Predicted: {err['pred_name']} ({err['pred_label']})\n")
            f.write("-" * 20 + "\n")

    print(f"✔ Saved {len(errors)} path of wrong file nel file txt.")
    '''

    os.makedirs(dir_path, exist_ok=True)
    # --- Save in CSV to visualize better.
    if len(errors) > 0:
        csv_path = f'{dir_path}/errors_fold_{fold + 1}.csv'

        keys = errors[0].keys()

        with open(csv_path, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(errors)

        print(f"✔ Dati CSV salvati in: {csv_path}")
    else:
        print("CSV empty.")
    return