import os, sys
import glob
import pandas as pd
import tqdm


# aws_prefix = 'https://graspingfield.s3-us-west-2.amazonaws.com/ho3d_fixed_render'
# gf_img_source = '/media/korrawe/data/works/gf/hp3d_fix/ho3d_correction/ho3d_fixed_render'

# /home/korrawe/halo_vae/exp/amt
# https://graspgen.s3-us-west-2.amazonaws.com/baseline/binoculars_000/0.png
img_source = '/home/korrawe/halo_vae/exp/amt'
aws_prefix = 'https://graspgen.s3-us-west-2.amazonaws.com/'
# baseline, grabnet, interloss

# 3 exp
# baseline - interloss
# interloss - grabnet
# interloss refine - grabnet refine

df = pd.DataFrame(columns=['image_1_1_url', 'image_1_2_url', 'image_1_3_url', 'image_1_4_url',
                           'image_2_1_url', 'image_2_2_url', 'image_2_3_url', 'image_2_4_url'])


left = 'baesline'
right = ''

object_list = ['binoculars', 'camera', 'fryingpan', 'mug', 'toothpaste', 'wineglass']
for object_type in object_list:
    # print()
    # obj_dir_l = os.path.join(aws_prefix, 'baseline')
    obj_dir_l = os.path.join(aws_prefix, 'interloss')
    # obj_dir_r = os.path.join(aws_prefix, 'interloss')
    obj_dir_r = os.path.join(aws_prefix, 'grabnet')

    n_sample = 20
    # for idx in range(n_sample):
    # Random left and right index between [0,20)

    # All-pair comparison
    for idx_l in range(n_sample):
        for idx_r in range(n_sample):
            row_dict = {}
            # sample_name_l = '%s_%03d' % (object_type, idx_l)
            sample_name_l = '%s_%03d_refine' % (object_type, idx_l)
            # sample_name_r = '%s_%03d' % (object_type, idx_r)
            sample_name_r = '%s_%03d_refine' % (object_type, idx_r)
            for i in range(4):
                # left
                image_l = os.path.join(obj_dir_l, sample_name_l, str(i) + '.png')
                # right
                image_r = os.path.join(obj_dir_r, sample_name_r, str(i) + '.png')

                print(image_l)
                print(image_r)
                print()
                row_dict['image_1_' + str(i+1) + '_url'] = image_l
                row_dict['image_2_' + str(i+1) + '_url'] = image_r
            df = df.append(row_dict, ignore_index=True)

# print(df)
outpath = '/home/korrawe/halo_vae/exp/amt/'
df.to_csv(outpath + 'inter_opt_grabnet_refine.csv', index=False)

## create URL for image
# exp_list1 = glob.glob(gf_img_source + "/*")

# # for instance in tqdm.tqdm(exp_list1):
# #         print(instance)
# #         for i in range(6):
# #                 # img_name = os.path.join(instance, '0_0.png')
# #                 suffix = '_gen' if i > 0 else ''
# #                 # if os.path.exists( os.path.join(instance, str(i) + '_0' + suffix + '.png')):
# #                 if os.path.exists( os.path.join(instance, str(i) + '_0' + suffix + '.png')):
# #                         img_base = os.path.join(instance, str(i) + '_')
# #                         img_base = img_base.replace(gf_img_source, aws_prefix)
# #                         row_dict = {}
# #                         # print("i = ", i)
# #                         for j in range(6):
# #                                 img_name_final = img_base + str(j) + suffix + '.png'
# #                                 row_dict['image_' + str(j+1) + '_url'] = img_name_final
# #                         df = df.append(row_dict ,ignore_index=True)
# #                         # print("exist")

# for instance in tqdm.tqdm(exp_list1):
#         print(instance)
#         if True:
#                 # img_name = os.path.join(instance, '0_0.png')
#                 if True:
#                         img_base = instance + '/' # os.path.join(instance, str(i) + '_')
#                         img_base = img_base.replace(gf_img_source, aws_prefix)
#                         row_dict = {}
#                         # print("i = ", i)
#                         for j in range(6):
#                                 img_name_final = img_base + str(j) + '.png'
#                                 row_dict['image_' + str(j+1) + '_url'] = img_name_final
#                         df = df.append(row_dict ,ignore_index=True)
#                         # print("exist")

# print(df.to_csv('/media/korrawe/data/works/gf/hp3d_fix/ho3d_correction/ho3d_fixed.csv', index=False))
