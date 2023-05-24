from skimage import segmentation,color
from scipy.stats import mode
from skimage.feature import  local_binary_pattern
import numpy as np
from tqdm import tqdm, trange
from tqdm.contrib import  tzip
from skimage.measure import regionprops
from skimage.feature import graycomatrix,  local_binary_pattern
import skimage
import cv2
from sklearn.preprocessing import MinMaxScaler
import copy

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
    neighbors = get_superpixel_neighbors(segments)
    for segment_index in unique_segments:
        # 获取当前超像素的掩码
        segment_mask = (segments == segment_index)

        # 获取当前超像素在LAB颜色空间中的像素值
        segment_lab_pixels = lab_image[segment_mask]
        segment_rgb_pixels = image[segment_mask]
        # 提取当前超像素的L值、A值、B值
        segment_L_values = np.mean(segment_lab_pixels[:, 0])
        segment_A_values = np.mean(segment_lab_pixels[:, 1])
        segment_B_values = np.mean(segment_lab_pixels[:, 2])
        segment_r_values = np.mean(segment_rgb_pixels[:, 0])
        segment_g_values = np.mean(segment_rgb_pixels[:, 1])
        segment_b_values = np.mean(segment_rgb_pixels[:, 2])
        # 计算当前超像素的颜色方差
        segment_color_variance = np.var(segment_lab_pixels, axis=0)
        segment_rgb_variance = np.var(segment_rgb_pixels, axis=0)
        feature_list =[]
        feature_list.append(segment_L_values)
        feature_list.append(segment_A_values)
        feature_list.append(segment_B_values)
        # feature_list.append(segment_r_values)
        # feature_list.append(segment_g_values)
        # feature_list.append(segment_b_values)
        feature_list.append(segment_color_variance[0])
        feature_list.append(segment_color_variance[1])
        feature_list.append(segment_color_variance[2])
        # feature_list.append(segment_rgb_variance[0])
        # feature_list.append(segment_rgb_variance[1])
        # feature_list.append(segment_rgb_variance[2])
        feature_array = feature_list
        features_list.append(feature_array)
    se_features_list = []
    for label,neighbor in neighbors.items():
        n_features = np.array(features_list[label-1],dtype=object)
        # print(n_features.shape)
        for n in neighbor:
            # print(np.array(features_list[n-1],dtype=object).shape)
            n_features += np.array(features_list[n-1],dtype=object)

        n_features /= len(neighbor)
        n_features = n_features.tolist()
        se_features_list.append(features_list[label-1]+n_features)
    return np.array(se_features_list,dtype=object)


def get_superpixel_neighbors(segments):

    neighbors = {}
    for label in np.unique(segments):
        mask = segments == label
        boundaries = segmentation.find_boundaries(mask, mode='outer')
        neighbor_labels = np.unique(segments[boundaries])
        nlist = []
        for neighbor_label in neighbor_labels:
            if neighbor_label != label:
                nlist.append(neighbor_label)
        neighbors[label] = nlist

    return neighbors

def single_feature_scaler(feature):
    scaler = MinMaxScaler()
    feature = scaler.fit_transform(feature.reshape(-1, 1))
    feature = feature.reshape(1, -1)[0]
    return feature


def GLCM_Feature(gray_array,
                 feature_name=['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'],
                 angle=[0, np.pi / 2]):
    glcm = graycomatrix(gray_array, [1], angle, 256, symmetric=True,
                        normed=True)  # , np.pi / 4, np.pi / 2, np.pi * 3 / 4

    # print(glcm.shape)
    texture_feature = []
    if 'entropy' in feature_name:
        feature_name.remove('entropy')
        P = copy.deepcopy(glcm)
        # normalize each GLCM
        P = P.astype(np.float64)
        glcm_sums = np.apply_over_axes(np.sum, P, axes=(0, 1))
        glcm_sums[glcm_sums == 0] = 1
        P /= glcm_sums
        entropy = np.apply_over_axes(np.sum, (P * -np.log(P)), axes=(0, 1))[0, 0]

        texture_feature.append(entropy[0])

    texture_feature = []
    for prop in feature_name:
        f = skimage.feature.graycoprops(glcm, prop)

        texture_feature.append(f[0])


    return np.array(texture_feature)


def slic_process2(img, segments, use_irregular_GLCM=True, region_size=20):
    feature_name = ['homogeneity', 'correlation']
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    r = 3  # 邻域半径
    p = 8 * r  # 邻域采样点数量
    uniformLBP_img = local_binary_pattern(gray_img, p, r, method="uniform")

    slic0 = cv2.ximgproc.createSuperpixelSLIC(img, region_size=region_size, algorithm=cv2.ximgproc.SLICO)

    slic0.iterate(10)  # 迭代次数，越大效果越好
    slic0.enforceLabelConnectivity()

    row, col = segments.shape[0], segments.shape[1]

    regions = regionprops(segments, gray_img)
    # 旋转不变均匀LBP的超像素块
    regions_LBP = regionprops(segments, uniformLBP_img)



    # adjoin_sp = []
    Superpixels_glcm = []
    Superpixels_lab = []
    Superpixels_lbp = []
    Superpixels_rgb = []
    Superpixels_context = []
    avg_lab = []
    avg_rgb = []

    print('颜色纹理信息：')
    for i, (props, props_LBP) in enumerate(tzip(regions, regions_LBP)):
        # 超像素的外接圆，圆心坐标为相对值
        (CircleX, CircleY), radius = cv2.minEnclosingCircle(props['Coordinates'])

        # 超像素内的所有点坐标
        coor = props['Coordinates']
        c_row = coor[:, 0:1].reshape(1, -1)[0]
        c_col = coor[:, 1:2].reshape(1, -1)[0]

        # 按segments的id顺序，计算每个超像素的特征信息

        # LAB直方图
        regions_pixels = lab_img[c_row, c_col]
        l_pixels = regions_pixels[:, 0]
        a_pixels = regions_pixels[:, 1]
        b_pixels = regions_pixels[:, 2]
        l_hist, _ = np.histogram(l_pixels, range=(0, 256), bins=16)
        l_hist = single_feature_scaler(l_hist)
        a_hist, _ = np.histogram(a_pixels, range=(0, 256), bins=16)
        a_hist = single_feature_scaler(a_hist)
        b_hist, _ = np.histogram(b_pixels, range=(0, 256), bins=16)
        b_hist = single_feature_scaler(b_hist)
        Superpixels_lab.append(np.concatenate((l_hist, a_hist, b_hist), axis=0))

        # RGB直方图
        regions_pixels = img[c_row, c_col]
        b_pixels = regions_pixels[:, 0]
        g_pixels = regions_pixels[:, 1]
        r_pixels = regions_pixels[:, 2]
        b_hist, _ = np.histogram(b_pixels, range=(0, 256), bins=16)
        g_hist, _ = np.histogram(g_pixels, range=(0, 256), bins=16)
        r_hist, _ = np.histogram(r_pixels, range=(0, 256), bins=16)
        b_hist = single_feature_scaler(b_hist)
        g_hist = single_feature_scaler(g_hist)
        r_hist = single_feature_scaler(r_hist)
        Superpixels_rgb.append(np.concatenate((b_hist, g_hist, r_hist), axis=0))

        # LBP直方图
        coor = props_LBP['Coordinates']
        c_row = coor[:, 0:1].reshape(1, -1)[0]
        c_col = coor[:, 1:2].reshape(1, -1)[0]
        regions_LBP = lab_img[c_row, c_col]
        lbp_hist, lbp_bins = np.histogram(regions_LBP)
        lbp_hist = single_feature_scaler(lbp_hist)
        Superpixels_lbp.append(lbp_hist)

        # 平均lab
        sp_lab = lab_img[c_row, c_col]
        avg_l = np.average(sp_lab[:, 0])
        avg_a = np.average(sp_lab[:, 1])
        avg_b = np.average(sp_lab[:, 2])
        avg_lab.append([avg_l, avg_a, avg_b])

        # 平均lab
        sp_bgr = img[c_row, c_col]
        avg_b = np.average(sp_lab[:, 0])
        avg_g = np.average(sp_lab[:, 1])
        avg_r = np.average(sp_lab[:, 2])
        avg_rgb.append([avg_r, avg_g, avg_b])

        if use_irregular_GLCM == True:
            # 超像素的外接矩形，不属于目标超像素的像素点为0
            gray_array = props.intensity_image
        else:
            gray_array = props.image
        texture_feature = GLCM_Feature(gray_array, feature_name)
        texture_feature = texture_feature.reshape(1, -1)[0]
        texture_feature = single_feature_scaler(texture_feature)
        Superpixels_glcm.append(texture_feature)

    Superpixels_lab = np.array(Superpixels_lab)
    Superpixels_lbp = np.array(Superpixels_lbp)
    Superpixels_glcm = np.array(Superpixels_glcm)
    Superpixels_rgb = np.array(Superpixels_rgb)
    avg_lab = np.array(avg_lab)
    avg_rgb = np.array(avg_rgb)

    print('上下文信息：')
    adjoin_sp = find_neighbor(segments)

    for i, ad in enumerate(tqdm(adjoin_sp)):

        ad = np.array(list(ad))

        ad_lab = Superpixels_lab[ad - 1]


        ad_glcm = Superpixels_glcm[ad - 1]


        all_rgb = np.average(ad_lab, axis=0)
        all_glcm = np.average(ad_glcm, axis=0)

        Superpixels_context.append(np.concatenate((all_rgb, all_glcm), axis=0))

    X_raw = np.concatenate((np.array(Superpixels_lab), np.array(Superpixels_lbp),
                            np.array(Superpixels_glcm), np.array(Superpixels_context)), axis=1)

    scaler = MinMaxScaler()
    X_raw = scaler.fit_transform(X_raw)

    return X_raw


def find_neighbor(segments):
    direction_x = np.array([-1, 0, 1, 0])
    direction_y = np.array([0, -1, 0, 1])
    adjoin_sp = [set() for i in range(segments.max())]

    max_x = segments.shape[0]
    max_y = segments.shape[1]
    for i in trange(max_x):
        for j in range(max_y):
            label = segments[i][j]
            x = i + direction_x
            y = j + direction_y

            for m in range(x.shape[0]):
                if x[m] > 0 and y[m] > 0 and x[m] < max_x and y[m] < max_y:
                    ix = x[m]
                    iy = y[m]
                    nlabel = segments[ix][iy]

                    if nlabel != label:
                        adjoin_sp[label - 1].add(nlabel - 1)


    return adjoin_sp
