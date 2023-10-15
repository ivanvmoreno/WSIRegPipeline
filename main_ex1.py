import os
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

import utils
import paths

import deephistreg as dhr

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### Identity - to create results without any registration (useful for creating the baseline)
# output_path = None # TO DEFINE
# dhr_params = dict()
# dhr_params['segmentation_mode'] = "deep_segmentation"
# dhr_params['initial_rotation'] = False
# dhr_params['affine_registration'] = False
# dhr_params['nonrigid_registration'] = False
# segmentation_params = dict()
# segmentation_params['model_path'] =  None # Path to segmentation model
# dhr_params['segmentation_params'] = segmentation_params
# load_masks = False
###

### Seg + Rotation Params - to create results from the initial rotation (useful for creating the affine training dataset)
# output_path = None # TO DEFINE
# dhr_params = dict()
# dhr_params['segmentation_mode'] = "deep_segmentation"
# dhr_params['initial_rotation'] = True
# dhr_params['affine_registration'] = False
# dhr_params['nonrigid_registration'] = False
# initial_rotation_params = dict()
# initial_rotation_params['angle_step'] = 1
# dhr_params['initial_rotation_params'] = initial_rotation_params
# segmentation_params = dict()
# segmentation_params['model_path'] =  None # Path to segmentation model
# dhr_params['segmentation_params'] = segmentation_params
# load_masks = False
###

## Seg + Rotation Params + Affine - to create results from the initial rotation + affine registration (useful for creating the nonrigid training dataset)
# STEP 1
# dataset_path = "/data/ANHIR_Parsed_1024_Masks"
# output_path = "/data/ANHIR_Out_Aff_1024_Masks_TRANSFORMED"
# dhr_params = dict()
# dhr_params["segmentation_mode"] = None  # or "deep_segmentation"
# dhr_params["initial_rotation"] = False
# dhr_params["affine_registration"] = True
# dhr_params["nonrigid_registration"] = False
# initial_rotation_params = dict()
# initial_rotation_params["angle_step"] = 1
# dhr_params["initial_rotation_params"] = initial_rotation_params
# affine_registration_params = dict()
# models_path = paths.models_path
# affine_registration_params["model_path"] = "/data/outdoor_megadepth.ckpt"
# affine_registration_params[
#     "main_config_path"
# ] = "./networks/feature_matching/configs/loftr/outdoor/loftr_ds_quadtree.py"
# affine_registration_params["affine_type"] = "quadtree"
# affine_registration_params["resize"] = False
# dhr_params["affine_registration_params"] = affine_registration_params
# segmentation_params = dict()
# segmentation_params["model_path"] = None  # Path to segmentation model
# dhr_params["segmentation_params"] = segmentation_params
# load_masks = True
##

# STEP 2 (only deformable)
dataset_path = "/data/ANHIR_Parsed_1024_Masks"
output_path = "/data/ANHIR_Out_Aff_1024_NEW_FIELD"
dhr_params = dict()
dhr_params["segmentation_mode"] = None  # or "deep_segmentation"
dhr_params["initial_rotation"] = False
dhr_params["affine_registration"] = True
dhr_params["nonrigid_registration"] = False
initial_rotation_params = dict()
initial_rotation_params["angle_step"] = 1
dhr_params["initial_rotation_params"] = initial_rotation_params
affine_registration_params = dict()
models_path = paths.models_path
affine_registration_params["model_path"] = "/data/outdoor_megadepth.ckpt"
affine_registration_params[
    "main_config_path"
] = "./networks/feature_matching/configs/loftr/outdoor/loftr_ds_quadtree.py"
affine_registration_params["affine_type"] = "quadtree"
affine_registration_params["resize"] = False
dhr_params["affine_registration_params"] = affine_registration_params
segmentation_params = dict()
segmentation_params["model_path"] = None  # Path to segmentation model
dhr_params["segmentation_params"] = segmentation_params
load_masks = True

# ### Seg + Rotation Params + Affine + Nonrigid - to create final nonrigid results
# output_path = None # TO DEFINE
# dhr_params = dict()
# dhr_params['segmentation_mode'] = "deep_segmentation"
# dhr_params['initial_rotation'] = True
# dhr_params['affine_registration'] = True
# dhr_params['nonrigid_registration'] = True
# initial_rotation_params = dict()
# initial_rotation_params['angle_step'] = 1
# dhr_params['initial_rotation_params'] = initial_rotation_params
# affine_registration_params = dict()
# models_path = None # TO DEFINE
# affine_registration_params['model_path'] = None # TO DEFINE
# affine_registration_params['affine_type'] = "simple"
# dhr_params['affine_registration_params'] = affine_registration_params
# nonrigid_registration_params = dict() # Params used during training
# nonrigid_registration_params['stride'] = 128
# nonrigid_registration_params['patch_size'] = (256, 256)
# nonrigid_registration_params['number_of_patches'] = 32
# nonrigid_registration_params['num_levels'] = 3
# nonrigid_registration_params['inner_iterations_per_level'] = [3, 3, 3]
# nonrigid_registration_params['model_path'] = None # TO DEFINE
# dhr_params['nonrigid_registration_params'] = nonrigid_registration_params
# segmentation_params = dict()
# segmentation_params['model_path'] =  None # TO DEFINE
# dhr_params['segmentation_params'] = segmentation_params
# load_masks = False
# ###


def run():
    show = False
    ids = range(0, 481)
    for current_id in ids:
        b_loading = time.time()
        current_pair = str(current_id)
        if load_masks:
            (
                source,
                target,
                source_landmarks,
                target_landmarks,
                status,
                source_mask,
                target_mask,
            ) = utils.load_pair(current_pair, dataset_path, load_masks=load_masks)
        else:
            (
                source,
                target,
                source_landmarks,
                target_landmarks,
                status,
            ) = utils.load_pair(current_pair, dataset_path, load_masks=load_masks)
        print("Current pair: ", current_pair)
        print("Status: ", status)
        source = torch.from_numpy(source).to(device)
        target = torch.from_numpy(target).to(device)

        if load_masks:
            source_mask = torch.from_numpy(source_mask).to(device)
            target_mask = torch.from_numpy(target_mask).to(device)

        e_loading = time.time()
        loading_time = e_loading - b_loading
        print("Time for loading and memory transfer: ", loading_time)
        b_registration = time.time()
        target, source, transformed_target, displacement_field, _ = dhr.deephistreg(
            target, source, device, dhr_params
        )

        e_registration = time.time()
        registration_time = e_registration - b_registration
        print("Time for registration: ", registration_time)

        transformed_source_landmarks = utils.transform_landmarks(
            source_landmarks, displacement_field
        )

        if load_masks:
            transformed_target_mask_save_path = os.path.join(
                output_path, current_pair, "transformed_target_mask.mha"
            )
            source_mask_save_path = os.path.join(
                output_path, current_pair, "source_mask.mha"
            )
            target_mask_save_path = os.path.join(
                output_path, current_pair, "target_mask.mha"
            )

            transformed_target_mask = utils.warp_tensor(
                target_mask, displacement_field, device=device
            )
            transformed_target_mask = transformed_target_mask.clamp(0, 1)
            transformed_target_mask = transformed_target_mask.cpu().numpy()
            blurred = cv2.GaussianBlur(transformed_target_mask, (5, 5), 0)
            transformed_target_mask = torch.from_numpy(blurred).to(device)
            transformed_target_mask = transformed_target_mask.round()

        source_save_path = os.path.join(output_path, current_pair, "source.mha")
        target_save_path = os.path.join(output_path, current_pair, "target.mha")
        transformed_target_save_path = os.path.join(
            output_path, current_pair, "transformed_target.mha"
        )
        source_landmarks_path = os.path.join(
            output_path, current_pair, "source_landmarks.csv"
        )
        transformed_source_landmarks_path = os.path.join(
            output_path, current_pair, "transformed_source_landmarks.csv"
        )
        displacement_field_save_path = os.path.join(
            output_path, current_pair, "transform_matrix.npy"
        )
        if status == "training":
            target_landmarks_path = os.path.join(
                output_path, current_pair, "target_landmarks.csv"
            )

        if not os.path.isdir(os.path.dirname(source_save_path)):
            os.makedirs(os.path.dirname(source_save_path))

        sitk.WriteImage(sitk.GetImageFromArray(source.cpu().numpy()), source_save_path)
        sitk.WriteImage(sitk.GetImageFromArray(target.cpu().numpy()), target_save_path)
        sitk.WriteImage(
            sitk.GetImageFromArray(transformed_target.cpu().numpy()),
            transformed_target_save_path,
        )
        utils.save_landmarks(source_landmarks, source_landmarks_path)
        utils.save_landmarks(
            transformed_source_landmarks, transformed_source_landmarks_path
        )
        np.save(displacement_field_save_path, displacement_field.cpu().numpy())
        if load_masks:
            sitk.WriteImage(
                sitk.GetImageFromArray(transformed_target_mask.cpu().numpy()),
                transformed_target_mask_save_path,
            )
            sitk.WriteImage(
                sitk.GetImageFromArray(source_mask.cpu().numpy()), source_mask_save_path
            )
            sitk.WriteImage(
                sitk.GetImageFromArray(target_mask.cpu().numpy()), target_mask_save_path
            )

        if status == "training":
            utils.save_landmarks(target_landmarks, target_landmarks_path)
            try:
                image_diagonal = np.sqrt(source.shape[0] ** 2 + source.shape[1] ** 2)
                rtre_initial = utils.calculate_rtre(
                    source_landmarks, target_landmarks, image_diagonal
                )
                rtre_final = utils.calculate_rtre(
                    transformed_source_landmarks, target_landmarks, image_diagonal
                )
                string_to_save = (
                    "Initial TRE: "
                    + str(np.median(rtre_initial))
                    + "\n"
                    + "Resulting TRE: "
                    + str(np.median(rtre_final))
                )
                txt_path = os.path.join(output_path, current_pair, "tre.txt")
                with open(txt_path, "w") as file:
                    file.write(string_to_save)
            except:
                pass
        time_to_save = str(registration_time + loading_time)
        time_txt_path = os.path.join(output_path, current_pair, "time.txt")
        with open(time_txt_path, "w") as file:
            file.write(time_to_save)


if __name__ == "__main__":
    run()
