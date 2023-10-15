import os


def find_missing_ids(output_dir, id_range):
    # getting all directories in the output directory
    dir_list = os.listdir(output_dir)

    # creating list of missing ids
    missing_ids = [
        str(directory) for directory in id_range if str(directory) not in dir_list
    ]

    return missing_ids


output_dir = "/data/ANHIR_Out_Aff_1024_Masks_TRANSFORMED_DEFORMABLE_EXP02"
id_range = range(0, 480)
missing_ids = find_missing_ids(output_dir, id_range)

print("Missing directories:", missing_ids)
