from skimage import io,segmentation,measure,color
import numpy as np
from scipy.stats import mode

def get_image_label(label_image):
    label_shape = label_image.shape
    # 创建一个空字典来存储颜色与类别的对应关系
    colors = {}
    # 遍历标签图像的每个像素
    for i in range(label_shape[0]):
        for j in range(label_shape[1]):
            # 获取当前像素的颜色值
            color = tuple(label_image[i, j])
            if color not in colors:
                colors[color] = 0
    num_kind = 0
    for color in colors.keys():
        num_kind += 1
        colors[color] = num_kind

    return colors



def superpixels_to_label(segments,label_image):
    # 获取超像素的唯一值
    colors= get_image_label(label_image)
    unique_segments = np.unique(segments)
    label_to_classify = []
    # 遍历每个超像素
    for segment_index in unique_segments:
        # 获取当前超像素的掩码
        segment_mask = (segments == segment_index)

        # 获取当前超像素的像素值
        label_sp = label_image[segment_mask]

        # 统计当前超像素中出现最多的颜色
        segment_color_mode = mode(label_sp, axis=0)

        # 获取众数对应的颜色值
        color = tuple(segment_color_mode.mode[0])
        label_to_classify.append( colors[color])
    return label_to_classify


def get_superpixels_feature(image,segments):
    # 获取超像素的唯一值
    unique_segments = np.unique(segments)
    lab_image = color.rgb2lab(image)
    # 遍历每个超像素
    features_list = []
    for segment_index in unique_segments:
        # 获取当前超像素的掩码
        segment_mask = (segments == segment_index)

        # 获取当前超像素在LAB颜色空间中的像素值
        segment_lab_pixels = lab_image[segment_mask]

        # 提取当前超像素的L值、A值、B值
        segment_L_values = np.mean(segment_lab_pixels[:, 0])
        segment_A_values = np.mean(segment_lab_pixels[:, 1])
        segment_B_values = np.mean(segment_lab_pixels[:, 2])

        # 计算当前超像素的颜色方差
        segment_color_variance = np.var(segment_lab_pixels, axis=0)

        feature_list =[]
        feature_list.append(segment_L_values)
        feature_list.append(segment_A_values)
        feature_list.append(segment_B_values)
        feature_list.append(segment_color_variance[0])
        feature_list.append(segment_color_variance[1])
        feature_list.append(segment_color_variance[2])
        feature_array = feature_list
        features_list.append(feature_array)
    return np.array(features_list,dtype=object)