
import pandas as pd
import numpy as np
import os
import shutil
import tifffile as tiff
import cv2
import SimpleITK as sitk
import pydicom as dcm


# 统一移动数据，并转uint8类型方便做超像素分割
def move_transform_data():
    data_path = r'E:\yyx\RanK\OriginData\data\train_z.xlsx'
    base_path = r'E:\yyx\RanK\OriginData\data\segdata'
    df = pd.read_excel(data_path)
    ids = df['id']
    labels = df['label']
    paths = df['path']

    for i in range(df.shape[0]):
        id = ids[i]
        label = labels[i]
        path = paths[i]
        file_name = path.split('\\')[-1]
        if id == '1557':
            # 移动数据至目标文件夹
            if label == 1:
                target_path = os.path.join(base_path, 'positive', id, 'original')
                temp_file_name = 'positive-' + str(id) + '-' + file_name
            else:
                target_path = os.path.join(base_path, 'negative', id, 'original')
                temp_file_name = 'negative-' + str(id) + '-' + file_name

            if not os.path.exists(target_path):
                os.makedirs(target_path, exist_ok=True)

            print(os.path.join(target_path, file_name))

            shutil.copy(path, os.path.join(target_path, file_name))
            shutil.copy(path, os.path.join(base_path, 'temp_test', temp_file_name))




def sort_index(x1, x2):
    row1, col1 = x1['row'], x1['col']
    row2, col2 = x2['row'], x2['col']
    if row1 != row2:
        return row1 < row2
    else:
        return col1 < col2

def get_min_index(mask_centroids, kidey_centroids):
    mask_point = np.array([mask_centroids[-1][0], mask_centroids[-1][1]])
    kidey_list = [[kidey_centroids[i][0], kidey_centroids[i][1]] for i in range(1, len(kidey_centroids))]

    # 元素平方和之后开根号
    dist = lambda x: np.linalg.norm(x - mask_point)
    match = min(kidey_list, key=dist)

    return kidey_list.index(match) + 1




# 数据加窗处理函数
def window_image(img, window_center, window_width, intercept, slope, rescale=True):
    img = (img * slope + intercept)  # for translation adjustments given in the dicom file.
    img_min = window_center - window_width // 2  # minimum HU level
    img_max = window_center + window_width // 2  # maximum HU level
    img[img < img_min] = img_min  # set img_min for all HU levels less than minimum HU level
    img[img > img_max] = img_max  # set img_max for all HU levels higher than maximum HU level
    if rescale:
        img = (img - img_min) / (img_max - img_min) * 255.0
    return img





def move_graph_pic():
    files = os.listdir(r'E:\yyx\RanK\OriginData\data\segdata\temp')

    for file in files:
        label = file.split('-')[0]
        id = file.split('-')[1]
        file_name = file.split('-')[2]

        src_path = os.path.join(r'E:\yyx\RanK\OriginData\data\segdata', label, id, 'uint8')

        shutil.copy(
            os.path.join(src_path, file_name + '_label.png'),
            os.path.join(r'E:\yyx\RanK\OriginData\data\segdata\graph-pic', label + '-' + id + '-' + file_name + '_label.png'))


def slide_patch():
    base_path = r'E:\yyx\RanK\OriginData\data\segdata\negative'
    data_path = r'E:\yyx\RanK\OriginData\data\PUCH'

    # patch大小
    patch_size = 40
    stride = 30
    threshold = 1 / (patch_size ** 2)


    ids = os.listdir(base_path)
    for id in ids:
        # 每次只处理一批数据
        if not id.startswith('BZ'):
            continue
        file_path = os.path.join(base_path, id, 'original')
        files = os.listdir(file_path)

        for file in files:


            # npy图像的保存路径
            npy_path = os.path.join(base_path, id, 'img_npy')
            if not os.path.exists(npy_path):
                os.makedirs(npy_path, exist_ok=True)
            # 原始tiff文件
            ori_img = tiff.imread(os.path.join(file_path, file))
            z = int(file.split('-')[4])

            label_img = sitk.ReadImage(os.path.join(data_path, 'negative', id.split('_')[0], 'A', id.split('_')[0] + '.nii'))

            # 将3D图像转换为3D数组
            label_numpy = sitk.GetArrayFromImage(label_img)
            tumor_mask = label_numpy[z, :, :].astype('uint8')
            # tumor_mask = label_numpy[z, :, :].astype('uint8')[::-1]

            # 肿瘤的mask
            tumor_mask = tumor_mask[
                         int(file.split('-')[0]):int(file.split('-')[1]),
                         int(file.split('-')[2]):int(file.split('-')[3])
                         ]
            print(id, z, tumor_mask.shape, ori_img.shape)
            # CT值转灰度值的斜率， intercept
            # intercept = -1024
            if ori_img.dtype == 'int16' and id != 'BZ175':
                intercept = 0
            else:
                intercept = -1024

            wc = 40
            wl = 70
            # 加窗之后的图像
            win_img = window_image(ori_img, wc, wl, intercept, 1, rescale=True)

            # 滑动窗口，切patch保存为npy
            for row in range((win_img.shape[0] - patch_size) // stride + 1):
                for col in range((win_img.shape[1] - patch_size) // stride + 1):

                    # patch的xy坐标
                    x_min = row * stride
                    y_min = col * stride

                    x_max = x_min + patch_size
                    if x_max > win_img.shape[0]:
                        x_max = win_img.shape[0]
                        x_min = x_max - patch_size
                    y_max = y_min + patch_size
                    if y_max > win_img.shape[1]:
                        y_max = win_img.shape[1]
                        y_min = y_max - patch_size

                    patch_tumor = tumor_mask[x_min:x_max, y_min:y_max]
                    patch_img = win_img[x_min:x_max, y_min:y_max]

                    if np.sum(patch_tumor) / (patch_size ** 2) > threshold:
                        # 当前patch含有tumor
                        label = 1
                    else:
                        label = 0

                    npy_name = "{}-{}-{}-{}-{}-{}-{}.npy".format(file.split('-')[-1].split('.')[0], label, x_min, x_max, y_min,
                                                              y_max, z)
                    # # 保存npy形式的图像
                    np.save(os.path.join(npy_path, npy_name), patch_img)





# 按照分割之后的图像的中心坐标进行剪裁上下左右保留5px的空余，北大三院的数据翻转，肿瘤医院不需要翻转
def cut_tiff():
    mask_path = r'D:\gyji\unet++,adda\test_data\merge\negative'
    data_path = r'E:\yyx\RanK\OriginData\data\PUCH\negative'
    save_path = r'E:\yyx\RanK\OriginData\data\segdata\negative'
    ids = os.listdir(data_path)
    for id in ids:
        label_img = sitk.ReadImage(os.path.join(data_path, id, 'A', id + '.nii'))
        # 将3D图像转换为3D数组
        label_numpy = sitk.GetArrayFromImage(label_img)
        label_C, label_H, label_W = label_numpy.shape
        print(id, label_C, label_H, label_W)
        x_mins = []
        x_maxs = []
        y_mins = []
        y_maxs = []
        # 选出裁剪的坐标
        for z in range(label_C):
            # 肿瘤的标签 北大三院数据需要翻转，肿瘤医院数据不需要翻转
            # z轴倒序
            # tumor_mask = label_numpy[z, :, :].astype('uint8')[::-1]
            tumor_mask = label_numpy[z, :, :].astype('uint8')
            max_value = np.max(tumor_mask)
            if (max_value) > 0:
                # 肿瘤肾脏合并之后的标签
                mask = cv2.imread(os.path.join(mask_path, id, id + "_slice_00%03d.png" % (z)))
                _, _, tumor_stats, tumor_centroids = cv2.connectedComponentsWithStats(tumor_mask)
                _, _, mask_stats, mask_centroids = cv2.connectedComponentsWithStats(mask[:, :, 0])

                # 选出左肾还是右肾有肿瘤，根据质心的距离进行计算
                min_index = get_min_index(tumor_centroids, mask_centroids)

                x_mins.append(mask_stats[min_index][1])
                x_maxs.append(mask_stats[min_index][1] + mask_stats[min_index][3])
                y_mins.append(mask_stats[min_index][0])
                y_maxs.append(mask_stats[min_index][0] + mask_stats[min_index][2])

        x_min = np.min(x_mins) - 5
        x_max = np.max(x_maxs) + 5
        y_min = np.min(y_mins) - 5
        y_max = np.max(y_maxs) + 5

        for z in range(label_C):
            # 肿瘤的标签 北大三院数据需要翻转，肿瘤医院数据不需要翻转
            # z轴倒序
            # tumor_mask = label_numpy[z, :, :].astype('uint8')[::-1]
            tumor_mask = label_numpy[z, :, :].astype('uint8')
            max_value = np.max(tumor_mask)
            if (max_value) > 0:

                with open(os.path.join(data_path, id, 'A', 'dcm_num.txt'), 'r', encoding='utf-8') as f:
                    dcm_num = int(f.read()) - 1 - z


                dcm_path = os.path.join(data_path, id, 'A', 'IM%d.DCM'%(dcm_num))
                data = dcm.dcmread(dcm_path)
                img = data.pixel_array

                img_cut = img[x_min: x_max, y_min: y_max]
                target_path = os.path.join(save_path, id, 'original')
                if not os.path.exists(target_path):
                    os.makedirs(target_path, exist_ok=True)
                # z轴命名
                tiff.imsave(os.path.join(target_path, "%d-%d-%d-%d-%d-IM%d.tif" % (x_min, x_max, y_min, y_max, z, dcm_num)), img_cut)




if __name__ == '__main__':
    slide_patch()
()



