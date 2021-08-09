import os
import numpy as np
import custom_scaler as cs

# single speaker
# data_dir = "../npy_data_mfcc39_landmark107_s19/train/"

# multi speaker
data_dir = "../npy_data_mfcc39_landmark107/train/"

if not os.path.exists(data_dir):
    print("data directory not exists. quitting..")
    quit()

# feature size
X_feature_size = 39
Y_feature_size = 107

# initialize  coefficients
X_data_min = np.array([float('inf')] * X_feature_size)
X_data_max = np.array([float('-inf')] * X_feature_size)
Y_data_min = np.array([float('inf')] * Y_feature_size)
Y_data_max = np.array([float('-inf')] * Y_feature_size)

for root, _, files in os.walk(data_dir):
    for file in files:
        if file.endswith("_X.npy"):
            X = np.load(os.path.join(root,file))
            for i in range(X_feature_size):
                X_min = X.flatten()[i::X_feature_size].min()
                X_max = X.flatten()[i::X_feature_size].max()
                if X_min < X_data_min[i]:
                    X_data_min[i] = X_min
                if X_max > X_data_max[i]:
                    X_data_max[i] = X_max
        if file.endswith("_Y.npy"):
            Y = np.load(os.path.join(root,file))
            for i in range(Y_feature_size):
                Y_min = Y.flatten()[i::Y_feature_size].min()
                Y_max = Y.flatten()[i::Y_feature_size].max()
                if Y_min < Y_data_min[i]:
                    Y_data_min[i] = Y_min
                if Y_max > Y_data_max[i]:
                    Y_data_max[i] = Y_max

X_scaler = cs.CustomScaler(X_feature_size)
Y_scaler = cs.CustomScaler(Y_feature_size)
X_scaler.fit_from_min_max_list(X_data_min, X_data_max, -0.7, 0.7)
Y_scaler.fit_from_min_max_list(Y_data_min, Y_data_max, -0.7, 0.7)

X_scaler.to_csv(data_dir + "x_scaler_coef.csv")
Y_scaler.to_csv(data_dir + "y_scaler_coef.csv")

# #----------------- testing----------------------#
# X_scaler = cs.CustomScaler(X_feature_size)
# Y_scaler = cs.CustomScaler(Y_feature_size)
# X_scaler.from_csv(data_dir + "x_scaler_coef.csv")
# Y_scaler.from_csv(data_dir + "y_scaler_coef.csv")


# X_data_min = np.array([float('inf')] * X_feature_size)
# X_data_max = np.array([float('-inf')] * X_feature_size)
# Y_data_min = np.array([float('inf')] * Y_feature_size)
# Y_data_max = np.array([float('-inf')] * Y_feature_size)


# for root, _, files in os.walk(data_dir):
#     for file in files:
#         if file.endswith("_X.npy"):
#             X = np.load(os.path.join(root,file))
#             X = X_scaler.scale(X)
#             for i in range(X_feature_size):
#                 X_min = X.flatten()[i::X_feature_size].min()
#                 X_max = X.flatten()[i::X_feature_size].max()
#                 if X_min <= X_data_min[i]:
#                     X_data_min[i] = X_min
#                 if X_max >= X_data_max[i]:
#                     X_data_max[i] = X_max
#         if file.endswith("_Y.npy"):
#             Y = np.load(os.path.join(root,file))
#             Y = Y_scaler.scale(Y)
#             for i in range(Y_feature_size):
#                 Y_min = Y.flatten()[i::Y_feature_size].min()
#                 Y_max = Y.flatten()[i::Y_feature_size].max()
#                 if Y_min <= Y_data_min[i]:
#                     Y_data_min[i] = Y_min
#                 if Y_max >= Y_data_max[i]:
#                     Y_data_max[i] = Y_max

# print("x min:", X_data_min, "x max:", X_data_max)
# print("y min:", Y_data_min, "y max:", Y_data_min)