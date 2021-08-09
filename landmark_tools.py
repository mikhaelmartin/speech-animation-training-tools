import pandas as pd

# labels
mfcc_labels = (
    ["mfcc_" + str(i) for i in range(14)[1:]]
    + ["delta_mfcc_" + str(i) for i in range(14)[1:]]
    + ["delta2_mfcc_" + str(i) for i in range(14)[1:]]
)

landmark_indexes = (
    [0, 1, 2, 3, 4, 5, 6, 7, 8]
    + [17, 18, 19, 20, 21]
    + [27, 28, 29, 30, 31, 32, 33]
    + [36, 37, 38, 39, 40, 41]
    + [48, 49, 50, 51]
    + [57, 58, 59, 60, 61, 62]
    + [66, 67]
)

mirror_pair_indexes = (
    [[0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9]]
    + [[17, 26], [18, 25], [19, 24], [20, 23], [21, 22]]
    + [[36, 45], [37, 44], [38, 43], [39, 42], [40, 47], [41, 46]]
    + [[31, 35], [32, 34]]
    + [[48, 54], [49, 53], [50, 52], [58, 56], [59, 55]]
    + [[60, 64], [61, 63], [67, 65]]
)

center_indexes = [8, 27, 28, 29, 30, 33, 51, 57, 62, 66]
landmark_labels = ["X_" + str(i) for i in landmark_indexes if i not in center_indexes]
landmark_labels += ["Y_" + str(i) for i in landmark_indexes]
landmark_labels += ["Z_" + str(i) for i in landmark_indexes]

all_landmark_labels = []
for axis in ["X_", "Y_", "Z_"]:
    for i in range(68):
        all_landmark_labels.append(axis + str(i))

def GetDisplacementLandmarkDF(y_data):
    # create all labels
    displacement_landmark_df = pd.DataFrame()
    for key in all_landmark_labels:
        displacement_landmark_df[key] = [0] * len(y_data)

    # left side landmarks
    for i in range(len(landmark_labels)):
        displacement_landmark_df[landmark_labels[i]] = y_data[:, i]

    # assign mirror data
    for left, right in mirror_pair_indexes:
        displacement_landmark_df["X_" + str(right)] = -displacement_landmark_df[
            "X_" + str(left)
        ]
        displacement_landmark_df["Y_" + str(right)] = displacement_landmark_df[
            "Y_" + str(left)
        ]
        displacement_landmark_df["Z_" + str(right)] = displacement_landmark_df[
            "Z_" + str(left)
        ]

    # make sure x value of center landmark is zero displacement
    for i in center_indexes:
        displacement_landmark_df["X_" + str(i)] = [0] * len(displacement_landmark_df)
    return displacement_landmark_df

# create displacement landmarks from only half face landmark displacement data 
def SaveDisplacementLandmark(save_file_path, y_data):
    # create all labels
    displacement_landmark_df = GetDisplacementLandmarkDF(y_data)
    displacement_landmark_df.to_csv(save_file_path, float_format="%.3f", index=False)


def GetAnimatedLandmarkDF(identity_csv, displacement_csv):
    identity_df = pd.read_csv(identity_csv)
    displacement_df = pd.read_csv(displacement_csv)
    animated_df = pd.DataFrame()
    for label in all_landmark_labels:
        animated_df[label] = identity_df.loc[0, label] + displacement_df[label]
    return animated_df

def SaveAnimatedLandmark(save_file_path, identity_csv, displacement_csv):    
    animated_df = GetAnimatedLandmarkDF(identity_csv, displacement_csv)    
    animated_df.to_csv(save_file_path, float_format="%.3f", index=False)
