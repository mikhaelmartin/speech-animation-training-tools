import os
import csv
import vpython as vp

input_mfcc_dir = "../mfcc/"

ground_def_file = open("../video/ground_def.csv")
openface_dir = "../openface_output/"
output_dir = (
    "../data/"
)
selected_mfcc_dir = output_dir+"0.mfcc/"
transformed_dir = output_dir + "1.transformed/"
symmetric_dir = output_dir + "2.symmetric/"
sliced_dir = output_dir + "3.sliced/"
ground_dir = output_dir + "4.ground/"
model_data_dir = output_dir + "5.model_data/"


def get_tranformed_data(row, headpos_index, pos_index):
    pos_row = [0] * 204
    pose_Tx = float(row[headpos_index])
    pose_Ty = float(row[headpos_index + 1])
    pose_Tz = float(row[headpos_index + 2])
    pose_Rx = float(row[headpos_index + 3])
    pose_Ry = float(row[headpos_index + 4])
    pose_Rz = float(row[headpos_index + 5])

    for i in range(68):
        # set origin to (0,0,0)
        x = float(row[pos_index + i]) - pose_Tx
        y = float(row[pos_index + i + 68]) - pose_Ty
        z = float(row[pos_index + i + 136]) - pose_Tz

        v = vp.vector(x, y, z)
        # set orientation facing the camera
        v = v.rotate(pose_Rx, vp.vector(-1, 0, 0))
        v = v.rotate(pose_Ry, vp.vector(0, -1, 0))
        v = v.rotate(pose_Rz, vp.vector(0, 0, -1))
        # somehow the output is inverted in y axis
        # so negative value is needed
        pos_row[i] = v.x
        pos_row[i + 68] = -v.y
        pos_row[i + 136] = v.z
    return pos_row


def copy_sym_vector(pos, destination_index, source_index):
    pos[destination_index] = -pos[source_index]
    pos[destination_index + 68] = pos[source_index + 68]
    pos[destination_index + 136] = pos[source_index + 136]


def get_symmetric_data(pos):
    copy = pos.copy()
    for i in range(68):
        if i >= 9 and i <= 16:
            copy_sym_vector(copy, i, 16 - i)
        elif i >= 22 and i <= 26:
            copy_sym_vector(copy, i, i - 2 * (i - 21) + 1)
        elif i >= 34 and i <= 35:
            copy_sym_vector(copy, i, i - 2 * (i - 33))
        elif i >= 42 and i <= 45:
            copy_sym_vector(copy, i, i - 2 * (i - 40) + 1)
        elif i >= 46 and i <= 47:
            copy_sym_vector(copy, i, i - 2 * (i - 43) + 1)
        elif i >= 52 and i <= 54:
            copy_sym_vector(copy, i, i - 2 * (i - 51))
        elif i >= 55 and i <= 56:
            copy_sym_vector(copy, i, i + 2 * (57 - i))
        elif i >= 63 and i <= 64:
            copy_sym_vector(copy, i, i - 2 * (i - 62))
        elif i == 65:
            copy_sym_vector(copy, i, 67)
        elif (
            i == 8
            or i == 33
            or i == 51
            or i == 57
            or i == 62
            or i == 65
            or (i >= 27 and i <= 30)
        ):
            copy[i] = 0
    return copy


def get_sliced_data(pos):
    return (
        # x
        pos[5:9]  # 0-3
        + pos[31:34]  # 4-6
        + pos[48:52]  # 7-10
        + pos[57:63]  # 11-16
        + pos[66:68]  # 17-18
        #  y
        + pos[73:77]  # [5 + 68: 9 + 68]
        + pos[99:102]  # [31 + 68: 34 + 68]
        + pos[116:120]  # [48 + 68: 52 + 68]
        + pos[125:131]  # [57 + 68: 63 + 68]
        + pos[134:136]  # [66 + 68: 68 + 68]
        #  z
        + pos[141:145]  # [5 + 136 : 9 + 136]
        + pos[167:170]  # [31 + 136 : 34 + 136]
        + pos[184:188]  # [48 + 136 : 52 + 136]
        + pos[193:199]  # [57 + 136 : 63 + 136]
        + pos[202:204]  # [66 + 136 : 68 + 136]
    )


def write_data_file(csv_file_loc, data):
    myFile = open(
        csv_file_loc,
        "w",
    )
    writer = csv.writer(myFile)
    writer.writerows(data)
    myFile.close()


def write_data_file_single_row(csv_file_loc, data_row):
    myFile = open(
        csv_file_loc,
        "w",
    )
    writer = csv.writer(myFile)
    writer.writerow(data_row)
    myFile.close()


if not os.path.exists(input_mfcc_dir):
    print("mfcc folder does not exists")
    exit()
if not os.path.exists(openface_dir):
    print("openface folder does not exists")
    exit()
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(selected_mfcc_dir):
    os.mkdir(selected_mfcc_dir)
if not os.path.exists(transformed_dir):
    os.mkdir(transformed_dir)
if not os.path.exists(symmetric_dir):
    os.mkdir(symmetric_dir)
if not os.path.exists(sliced_dir):
    os.mkdir(sliced_dir)
if not os.path.exists(ground_dir):
    os.mkdir(ground_dir)
if not os.path.exists(model_data_dir):
    os.mkdir(model_data_dir)


filename_list = [item for item in os.listdir(openface_dir) if (".csv" in item)]

# create ground frame dictionary
ground_dict = dict()
csv_reader = csv.reader(ground_def_file, delimiter=",")
next(csv_reader)
for row in csv_reader:
    key = str(row[0])
    ground_dict[key] = int(row[1])

# data processing
for file in filename_list:
    csv_mfcc = csv.reader(
            open(input_mfcc_dir + file[:-4]+"_mfcc.csv"), delimiter=",")
    mfcc_rows = list(list())
    for row in csv_mfcc:
        mfcc_rows.append(row)

    csv_openface = csv.reader(open(openface_dir + file), delimiter=",")
    row1 = next(csv_openface)
    success_index = row1.index("success")
    headpos_index = row1.index("pose_Tx")
    pos_index = row1.index("X_0")
    mfcc = list(list())
    transformed_pos = list(list())
    symmetric_pos = list(list())
    sliced_pos = list(list())
    ground_row = list()
    model_data_pos = list(list())
    frame_index = 0

    # get sliced position data and ground
    for index, row in enumerate(csv_openface):
        # print(float(row[confidence_index]))
        if float(row[success_index]) == 1:
            # print(mfcc_rows[index])
            mfcc.append(mfcc_rows[index])

            # get transformed data, facing the camera
            pos_row = get_tranformed_data(row, headpos_index, pos_index)

            # add transformed data
            transformed_pos.append(pos_row)

            # set landmark to be perfect symmetric
            symmetric_pos_row = get_symmetric_data(pos_row)

            # add symmetric data
            symmetric_pos.append(symmetric_pos_row)

            # get sliced pos. only sublist of symmetric_pos_row
            sliced_pos_row = get_sliced_data(symmetric_pos_row)

            # add sliced data
            sliced_pos.append(sliced_pos_row)

            # if this frame is the ground_truth
            if file[:-4] in ground_dict:
                if frame_index == ground_dict[file[:-4]]:
                    ground_row = sliced_pos_row
            else:
                if frame_index == 0:
                    ground_row = sliced_pos_row

            # next frame
            frame_index += 1

    # get model data
    for sliced_row in sliced_pos:
        model_data_row = list()
        for index in range(len(sliced_row)):
            model_data_row.append(
                str(float(sliced_row[index]) - float(ground_row[index]))
            )
        model_data_pos.append(model_data_row)

    # save files
    write_data_file(
        selected_mfcc_dir + file[:-4] + "_mfcc.csv", mfcc
    )

    write_data_file(
        transformed_dir + file[:-4] + "_pos.csv", transformed_pos
    )

    write_data_file(
        symmetric_dir + file[:-4] + "_symmetric.csv", symmetric_pos
    )

    write_data_file(sliced_dir + file[:-4] + "_sliced.csv", sliced_pos)

    write_data_file_single_row(
        ground_dir + file[:-4] + "_ground.csv", ground_row
    )

    write_data_file(
        model_data_dir + file[:-4] + "_model_data.csv", model_data_pos
    )

    # print("train_pos", len(training_pos[0]))
