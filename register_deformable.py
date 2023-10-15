import sys
import os
import time
import json

import utils
import networks.deformable as deformable

import torch
import numpy as np
import SimpleITK as sitk

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def register(current_id, dataset_path, output_path, load_masks):
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
        ) = utils.load_pair_registered(
            current_pair, dataset_path, load_masks=load_masks
        )
    else:
        (
            source,
            target,
            source_landmarks,
            target_landmarks,
            status,
        ) = utils.load_pair_registered(
            current_pair, dataset_path, load_masks=load_masks
        )
    print("Current pair: ", current_pair)
    print("Status: ", status)

    e_loading = time.time()
    loading_time = e_loading - b_loading
    print("Time for loading and memory transfer: ", loading_time)
    b_registration = time.time()

    try:
        displacement_field = deformable.register(
            source, target, source_mask, target_mask
        )
    except Exception as e:
        print(e)
        displacement_field = torch.zeros(
            (source.shape[0], source.shape[1], 2), dtype=torch.float32
        )

    source = torch.from_numpy(source).to(device)
    target = torch.from_numpy(target).to(device)

    transformed_target = utils.warp_tensor(target, displacement_field, device=device)

    transformed_source_landmarks = utils.transform_landmarks(
        source_landmarks, displacement_field
    )

    e_registration = time.time()
    registration_time = e_registration - b_registration
    print("Time for registration: ", registration_time)

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
    current_id = int(sys.argv[1])  # get current_id from command line arguments
    # Parse `args` from json
    args = json.loads(sys.argv[2])  # get args from command line arguments

    # You can now access args like this:
    dataset_path = args["dataset_path"]
    output_path = args["output_path"]
    load_masks = args["load_masks"]

    # Call your function
    register(current_id, dataset_path, output_path, load_masks)
