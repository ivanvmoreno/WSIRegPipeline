import cv2
import numpy as np
import SimpleITK as sitk
import torch


def check_image_values(image):
    print("Max value: ", np.max(image))
    print("Min value: ", np.min(image))
    print("Mean value: ", np.mean(image))
    print("Contains NaNs: ", np.isnan(image).any())
    print("Contains Infs: ", np.isinf(image).any())


def register(source, target, source_mask, target_mask, device="cuda"):
    # Inverting intensity values
    # target = 255 - target
    # source = 255 - source

    # Background Removal
    # target_mask = np.array(target_mask != 0, dtype=np.uint8)
    # source_mask = np.array(source_mask != 0, dtype=np.uint8)
    # target = cv2.bitwise_and(target, target, mask=target_mask)
    # source = cv2.bitwise_and(source, source, mask=source_mask)

    # Getting SimpleITK Images from numpy arrays
    # source_image_inv_sitk = sitk.GetImageFromArray(source)
    # target_image_inv_sitk = sitk.GetImageFromArray(target)

    # resampler = sitk.ResampleImageFilter()
    # resampler.SetSize(source_image_inv_sitk.GetSize())
    # resampler.SetOutputSpacing(source_image_inv_sitk.GetSpacing())
    # target_image_inv_sitk = resampler.Execute(target_image_inv_sitk)
    # source_image_inv_sitk = sitk.Cast(source_image_inv_sitk, sitk.sitkFloat32)
    # target_image_inv_sitk = sitk.Cast(target_image_inv_sitk, sitk.sitkFloat32)

    # Getting SimpleITK Images from numpy arrays
    source_image_inv_sitk = sitk.GetImageFromArray(source)
    target_image_inv_sitk = sitk.GetImageFromArray(target)

    # Define the transform
    transformDomainMeshSize = [3] * source_image_inv_sitk.GetDimension()
    tx = sitk.BSplineTransformInitializer(
        source_image_inv_sitk, transformDomainMeshSize
    )

    R = sitk.ImageRegistrationMethod()
    R.SetInitialTransformAsBSpline(tx, inPlace=True, scaleFactors=[1, 2, 4])
    R.SetMetricAsMattesMutualInformation(50)
    R.SetMetricSamplingStrategy(R.REGULAR)
    R.SetMetricSamplingPercentage(0.2)

    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])

    R.SetInterpolator(sitk.sitkBSpline)

    R.SetOptimizerAsGradientDescentLineSearch(
        0.5, 100, convergenceMinimumValue=1e-4, convergenceWindowSize=5
    )

    outTx = R.Execute(source_image_inv_sitk, target_image_inv_sitk)
    displacement_field_image = sitk.TransformToDisplacementField(
        outTx,
        sitk.sitkVectorFloat64,
        source_image_inv_sitk.GetSize(),
        source_image_inv_sitk.GetOrigin(),
        source_image_inv_sitk.GetSpacing(),
        source_image_inv_sitk.GetDirection(),
    )
    # Convert the displacement field image to a numpy array
    displacement_field_array = sitk.GetArrayFromImage(displacement_field_image)

    # Convert the numpy array to a PyTorch tensor and move to GPU
    return torch.from_numpy(displacement_field_array).to(device)
