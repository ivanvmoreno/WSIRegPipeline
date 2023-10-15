import os
import time

import utils
import paths

import cv2
import numpy as np
import pandas as pd
import scipy.ndimage as nd
import SimpleITK as sitk
from skimage import color
from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor
from tiatoolbox.tools.registration.wsi_registration import match_histograms


def parse_dataset(
    csv_path,
    dataset_path,
    output_path,
    segmentor,
    ids_to_process=None,
    output_max_size=1024,
):
    csv_file = pd.read_csv(csv_path)
    for current_case in csv_file.iterrows():
        current_id = current_case[1]["Unnamed: 0"]

        if ids_to_process is not None:
            if isinstance(ids_to_process, list):
                if current_id not in ids_to_process:
                    continue
            else:
                if current_id != ids_to_process:
                    continue

        size = current_case[1]["Image size [pixels]"]
        diagonal = int(current_case[1]["Image diagonal [pixels]"])
        y_size, x_size = int(size.split(",")[0][1:]), int(size.split(",")[1][:-1])
        source_path = current_case[1]["Source image"]
        target_path = current_case[1]["Target image"]
        source_landmarks_path = current_case[1]["Source landmarks"]
        target_landmarks_path = current_case[1]["Target landmarks"]
        status = current_case[1]["status"]
        start_time = time.time()
        print(f"Current case: {current_id} ({os.path.dirname(source_path)})")

        extension = source_path[-4:]

        source_path = os.path.join(dataset_path, source_path.replace(extension, ".mha"))
        target_path = os.path.join(dataset_path, target_path.replace(extension, ".mha"))
        source_landmarks_path = os.path.join(dataset_path, source_landmarks_path)
        target_landmarks_path = os.path.join(dataset_path, target_landmarks_path)

        # loading
        source_landmarks = utils.load_landmarks(source_landmarks_path)
        if status == "training":
            target_landmarks = utils.load_landmarks(target_landmarks_path)

        source = color.rgb2gray(sitk.GetArrayFromImage(sitk.ReadImage(source_path)))
        target = color.rgb2gray(sitk.GetArrayFromImage(sitk.ReadImage(target_path)))

        # preprocessing
        source, target = utils.preprocess_image(source), utils.preprocess_image(target)

        source, target = match_histograms(source, target)

        source = 1 - source
        target = 1 - target

        # source = 1 - utils.normalize(source)
        # target = 1 - utils.normalize(target)

        padded_source, padded_target = utils.pad_images_np(source, target)
        padded_source_landmarks = utils.pad_landmarks(
            source_landmarks, source.shape, padded_source.shape
        )
        if status == "training":
            padded_target_landmarks = utils.pad_landmarks(
                target_landmarks, target.shape, padded_target.shape
            )

        # resampling
        resample_factor = np.max(padded_source.shape) / output_max_size
        gaussian_sigma = resample_factor / 1.25

        smoothed_source = nd.gaussian_filter(padded_source, gaussian_sigma)
        smoothed_target = nd.gaussian_filter(padded_target, gaussian_sigma)

        resampled_source = utils.resample_image(smoothed_source, resample_factor)
        resampled_target = utils.resample_image(smoothed_target, resample_factor)
        resampled_source_landmarks = utils.resample_landmarks(
            padded_source_landmarks, resample_factor
        )
        if status == "training":
            resampled_target_landmarks = utils.resample_landmarks(
                padded_target_landmarks, resample_factor
            )

        to_save_source_mha = sitk.GetImageFromArray(
            (utils.normalize(resampled_source).astype(np.float32))
        )
        to_save_target_mha = sitk.GetImageFromArray(
            (utils.normalize(resampled_target).astype(np.float32))
        )
        to_save_source_jpg = sitk.GetImageFromArray(
            ((utils.normalize(resampled_source).astype(np.float32)) * 255).astype(
                np.ubyte
            )
        )
        to_save_target_jpg = sitk.GetImageFromArray(
            ((utils.normalize(resampled_target).astype(np.float32)) * 255).astype(
                np.ubyte
            )
        )
        to_save_source_landmarks = resampled_source_landmarks.astype(np.float32)
        if status == "training":
            to_save_target_landmarks = resampled_target_landmarks.astype(np.float32)

        sample_output_path = os.path.join(output_path, str(current_id))
        to_save_source_jpg_path = os.path.join(sample_output_path, "source.jpg")
        to_save_target_jpg_path = os.path.join(sample_output_path, "target.jpg")
        to_save_source_png_path = os.path.join(sample_output_path, "source.png")
        to_save_source_mask_png_path = os.path.join(
            sample_output_path, "source_mask.png"
        )
        to_save_source_mask_jpg_path = os.path.join(
            sample_output_path, "source_mask.jpg"
        )
        to_save_source_mask_mha_path = os.path.join(
            sample_output_path, "source_mask.mha"
        )
        to_save_source_png_path = os.path.join(sample_output_path, "source.png")
        to_save_target_mask_png_path = os.path.join(
            output_path, str(current_id), "target_mask.png"
        )
        to_save_target_mask_jpg_path = os.path.join(
            output_path, str(current_id), "target_mask.jpg"
        )
        to_save_target_mask_mha_path = os.path.join(
            output_path, str(current_id), "target_mask.mha"
        )
        to_save_target_png_path = os.path.join(sample_output_path, "target.png")
        to_save_source_mha_path = os.path.join(sample_output_path, "source.mha")
        to_save_target_mha_path = os.path.join(sample_output_path, "target.mha")
        to_save_source_landmarks_path = os.path.join(
            sample_output_path, "source_landmarks.csv"
        )
        if status == "training":
            to_save_target_landmarks_path = os.path.join(
                sample_output_path, "target_landmarks.csv"
            )

        if not os.path.isdir(os.path.dirname(to_save_source_mha_path)):
            os.makedirs(os.path.dirname(to_save_source_mha_path))

        sitk.WriteImage(to_save_source_mha, to_save_source_mha_path)
        sitk.WriteImage(to_save_target_mha, to_save_target_mha_path)
        sitk.WriteImage(to_save_source_jpg, to_save_source_jpg_path)
        sitk.WriteImage(to_save_target_jpg, to_save_target_jpg_path)
        sitk.WriteImage(to_save_source_jpg, to_save_source_png_path)
        sitk.WriteImage(to_save_target_jpg, to_save_target_png_path)
        utils.save_landmarks(to_save_source_landmarks, to_save_source_landmarks_path)
        if status == "training":
            utils.save_landmarks(
                to_save_target_landmarks, to_save_target_landmarks_path
            )

        # tissue segmentation
        output = segmentor.predict(
            [to_save_source_png_path, to_save_target_png_path],
            save_dir=os.path.join(sample_output_path, "mask"),
            mode="tile",
            resolution=1.0,
            units="baseline",
            patch_input_shape=[1024, 1024],
            patch_output_shape=[512, 512],
            stride_shape=[512, 512],
            on_gpu=True,
            crash_on_exception=True,
        )

        fixed_mask = np.load(output[0][1] + ".raw.0.npy")
        moving_mask = np.load(output[1][1] + ".raw.0.npy")

        fixed_mask = np.argmax(fixed_mask, axis=-1) == 2
        moving_mask = np.argmax(moving_mask, axis=-1) == 2

        fixed_mask = utils.post_processing_mask(fixed_mask).astype(np.uint8) * 255
        moving_mask = utils.post_processing_mask(moving_mask).astype(np.uint8) * 255

        cv2.imwrite(to_save_source_mask_png_path, fixed_mask)
        cv2.imwrite(to_save_source_mask_jpg_path, fixed_mask)
        cv2.imwrite(to_save_target_mask_png_path, moving_mask)
        cv2.imwrite(to_save_target_mask_jpg_path, moving_mask)
        sitk.WriteImage(
            sitk.GetImageFromArray(fixed_mask), to_save_source_mask_mha_path
        )
        sitk.WriteImage(
            sitk.GetImageFromArray(moving_mask), to_save_target_mask_mha_path
        )

        print(f"Processing time: {time.time() - start_time}")


def create_training_dataset(results_path, dataset_path, size, mode="min"):
    show = False
    ids = range(0, 481)
    for current_id in ids:
        current_id = str(current_id)
        case_path = os.path.join(results_path, current_id)
        transformed_target_path = os.path.join(case_path, "transformed_target.mha")
        source_path = os.path.join(case_path, "source.mha")

        transformed_target = sitk.GetArrayFromImage(
            sitk.ReadImage(transformed_target_path)
        )
        source = sitk.GetArrayFromImage(sitk.ReadImage(source_path))

        t_source = torch.from_numpy(source)
        t_target = torch.from_numpy(transformed_target)
        if mode == "min":
            new_shape = utils.calculate_new_shape_min(
                (t_source.size(0), t_source.size(1)), size
            )
            if min(new_shape) == min(source.shape):
                print("Resampling not required")
                resampled_source = t_source
                resampled_target = t_target
            else:
                resampled_source = utils.resample_tensor(t_source, new_shape)
                resampled_target = utils.resample_tensor(t_target, new_shape)
        elif mode == "max":
            new_shape = utils.calculate_new_shape_max(
                (t_source.size(0), t_source.size(1)), size
            )
            if max(new_shape) == max(source.shape):
                print("Resampling not required")
                resampled_source = t_source
                resampled_target = t_target
            else:
                resampled_source = utils.resample_tensor(t_source, new_shape)
                resampled_target = utils.resample_tensor(t_target, new_shape)

        transformed_target_resampled = resampled_target.numpy()
        source_resampled = resampled_source.numpy()

        target_landmarks_path = os.path.join(case_path, "target_landmarks.csv")
        try:
            target_landmarks = utils.load_landmarks(target_landmarks_path)
            status = "training"
        except:
            status = "evaluation"

        print("Current ID: ", current_id)
        print("Transformed target shape: ", transformed_target.shape)
        print("Source shape: ", source.shape)
        print("Resampled target shape: ", transformed_target_resampled.shape)
        print("Resampled source shape: ", source_resampled.shape)

        if show:
            plt.figure()
            plt.subplot(2, 2, 1)
            plt.imshow(source, cmap="gray")
            plt.axis("off")
            plt.subplot(2, 2, 2)
            plt.imshow(transformed_target, cmap="gray")
            plt.axis("off")
            plt.subplot(2, 2, 3)
            plt.imshow(source_resampled, cmap="gray")
            plt.axis("off")
            plt.subplot(2, 2, 4)
            plt.imshow(transformed_target_resampled, cmap="gray")
            plt.axis("off")
            plt.show()

        to_save_source = sitk.GetImageFromArray(transformed_target_resampled)
        to_save_target = sitk.GetImageFromArray(source_resampled)

        output_path = os.path.join(dataset_path, status, current_id)
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        source_output_path = os.path.join(output_path, "source.mha")
        target_output_path = os.path.join(output_path, "target.mha")
        sitk.WriteImage(to_save_source, source_output_path)
        sitk.WriteImage(to_save_target, target_output_path)


if __name__ == "__main__":
    segmentor = SemanticSegmentor(
        pretrained_model="unet_tissue_mask_tsef",
        num_loader_workers=4,
        batch_size=4,
    )

    csv_path = "/data/dataset_medium.csv"
    dataset_path = "/data/ANHIR_MHA"
    output_path = "/data/ANHIR_Parsed_1024_Masks"
    output_max_size = 1024
    ids_to_process = list(range(300, 310))
    parse_dataset(
        csv_path, dataset_path, output_path, segmentor, ids_to_process, output_max_size
    )

    # The purpose of the code below is to create training dataset for the next registration step (e.g. from rotation alignment to affine or from affine to nonrigid)
    # results_path = None
    # dataset_path = None
    # size = 1024 # Max shape in the training dataset (useful for e.g. decreasing the resolution for initial rotation search or affine registration)
    # create_training_dataset(results_path, dataset_path, size, "min")
