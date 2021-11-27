import pandas as pd
import shutil
import os

celeb_a_fpath = 'data/CelebA'
df = pd.read_csv(f'{celeb_a_fpath}/list_attr_celeba.txt', sep = '\s+').rename_axis('fname').reset_index()
df.replace({-1:0}, inplace=True)

# d = df.iloc[:,1:].sum(0).reset_index()
# d[0] = d[0]/len(df)
# print(d.set_index('index').sort_values(0).to_markdown())
#

#['Eyeglasses','Mustache', 'Heavy_Makeup', 'Gray_Hair', 'Bald']
for attribute in ['Eyeglasses','Mustache', 'Heavy_Makeup', 'Gray_Hair', 'Bald', 'Bushy_Eyebrows', 'Bangs']:
    print(attribute)
    attribute_flist = df.fname[df[attribute] == 1].to_list()

    attribute_dir = f'data/CelebA_{attribute}'
    attribute_class_dir = f'{attribute_dir}/0'
    os.makedirs(attribute_class_dir)

    for f in attribute_flist:
        shutil.copy(f'{celeb_a_fpath}/img_align_celeba/{f}', attribute_class_dir)

    resample_cmdline = f"python3 resize_training_subset.py -d_in {attribute_dir} -s 64 -d_out data/CelebA_{attribute}_size_64 --sample_grid_fname CelebA_{attribute}_grid.jpg -N 100"

    os.system(resample_cmdline)

    get_dd_cmdline = f'python lpips_2dir_allpairs.py -d0 data/CelebA_samples_size_64/ -d1 data/CelebA_{attribute}_size_64 -o distances/celebA_CelebA_{attribute}.txt -N 20'
    os.system(get_dd_cmdline)
