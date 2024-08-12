import numpy as np
import cv2

def get_center(i_array):
    n_row, n_col = i_array.shape
    x_coords = np.repeat(i_array[:, ::2], 2).reshape((-1, n_col))
    y_coords = np.repeat(i_array[:, 1::2], 2).reshape((-1, n_col))

    x_center = (np.max(x_coords) + np.min(x_coords))/2
    y_center = (np.max(y_coords) + np.min(y_coords))/2

    center = np.array((x_center, y_center))

    return x_coords, y_coords, center


def geometry_rotation(i_array, i_degree, center_mode=True):
    i_array_2 = i_array.copy()
    n_row, n_col = i_array_2.shape

    _, _, center = get_center(i_array)
    print(f'center coords = {center}')

    # radians
    i_radian = np.radians(i_degree)
    
    for i in range(0, n_col, 2):
        # x_rotated = x * cos - y * sin
        # y_rotated = x * cos + y * sin
        
        # x
        i_array_2[:, i] = ((i_array[:, i] - center[0]) * np.cos(i_radian)
                            - (i_array[:, i+1] - center[1]) * np.sin(i_radian)) + center[0]
        # y
        i_array_2[:, i+1] = ((i_array[:, i] - center[0]) * np.cos(i_radian)
                            + (i_array[:, i+1] - center[1]) * np.sin(i_radian)) + center[1]

    return i_array_2


# def geometry_rotation(i_array, i_degree, center_mode=True):
#     i_array_2 = i_array.copy()
#     n_row, n_col = i_array_2.shape

#     if center_mode:
#         x_coords, y_coords, center = get_center(i_array_2)
#         print(f'center coords = {center}')
#     else:
#         x_coords, y_coords, _ = get_center(i_array_2)
#         center = (n_row//2, n_col//2)
#         print(f'center coords = {center}')
#     i_radian = np.radians(i_degree)

#     rotation_matrix = cv2.getRotationMatrix2D(center, i_radian, 1.0)
    
#     rotated_x = cv2.warpAffine(x_coords, rotation_matrix, x_coords.shape)
#     rotated_y = cv2.warpAffine(y_coords, rotation_matrix, y_coords.shape)

#     i_array_2[:, ::2] = rotated_x[:, ::2]
#     i_array_2[:, 1::2] = rotated_y[:, 1::2]

#     return i_array_2


def getstd(i_array):
    temp = np.std(i_array, axis=0)

    return temp

def get_noise(i_array, i_std):
    array_2 = np.zeros(shape=i_array.shape)
    n_row, n_col = array_2.shape
    for i in range(n_col):
        array_2[:, i] = np.random.normal(loc=0.0,
                                         scale=i_std[i],
                                         size=n_row)
    return array_2

def add_noise(i_array, noise, alpha=0.1):
    array_2 = i_array + alpha * noise

    return array_2

# 
def noised_array(i_array, alpha=0.1):
    i_std = getstd(i_array)
    noise = get_noise(i_array, i_std)

    array_2 = i_array + alpha * noise

    return array_2


def nan_drop(i_array, nan_ratio=0.05):
    array_2 = i_array.copy()
    # array_2 = np.zeros(shape=i_array.shape)
    n_row, n_col = array_2.shape
    for i in range(0, n_col-1, 2):
        mask = np.random.rand(n_row) < nan_ratio
        array_2[:, i][mask] = np.nan
        array_2[:, i+1][mask] = np.nan
        # array_2[:, i][mask] = -1
        # array_2[:, i+1][mask] = -1

    return array_2


def augment_df(df, label_df, degree, mode=1, center_mode=True):
    if mode==0:
        df_out = df.copy()
        label_out = label_df.copy()

        return df_out, label_out
    
    if mode==1:
        df_0 = df.copy()
        df_1 = geometry_rotation(df.copy(), degree, center_mode=center_mode)

        df_out = np.concatenate((df_0, df_1), axis=0)
        label_out = np.concatenate((label_df, label_df), axis=0)

        return df_out, label_out
    
    if mode==2:
        df_0 = df.copy()
        df_1 = geometry_rotation(df.copy(), degree, center_mode=center_mode)
        df_2 = noised_array(df.copy(), 0.1)

        df_out = np.concatenate((df_0, df_1, df_2), axis=0)
        label_out = np.concatenate((label_df, label_df, label_df), axis=0)

        return df_out, label_out
    
    if mode==3:
        df_0 = df.copy()
        df_1 = geometry_rotation(df.copy(), degree, center_mode=center_mode)
        df_2 = noised_array(df.copy(), 0.1)
        df_3 = nan_drop(df.copy())

        df_out = np.concatenate((df_0, df_1, df_2, df_3), axis=0)
        label_out = np.concatenate((label_df, label_df, label_df, label_df), axis=0)

        return df_out, label_out
    else:
        print('[Error] Mode 1~3 only available')
