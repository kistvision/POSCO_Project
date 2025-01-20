import os

#T91
#BSDS100
data_name = 'T91+1_zoom'
# Prepare dataset
os.system(f"python3 ./prepare_dataset.py --images_dir ../data/{data_name}/original --output_dir ../data/{data_name}/ESPCN/train --image_size 70 --step 35 --num_workers 16")

# Split train and valid
os.system(f"python3 ./split_train_valid_dataset.py --train_images_dir ../data/{data_name}/ESPCN/train --valid_images_dir ../data/{data_name}/ESPCN/valid --valid_samples_ratio 0.1")
