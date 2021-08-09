import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


key = "Y_57"
lower = 0
upper = 50
true_dir = "/home/alkhemi/Documents/thesis/true_smallset"
prediction_dir = "/home/alkhemi/Documents/thesis/prediction_smallset"
data_type = "test"
speaker = "s31"
file = "pgia6a"

plots = [
    [
        "True",
        f"{true_dir}/{data_type}/{speaker}/{file}_dis.csv",
    ],
    # [
    #     "MLP",
    #     f"{prediction_dir}/MLP_u64_l3_seq11-2-11-2_multi/direct/{data_type}/{speaker}/{file}_dis_pred.csv",
    # ],
    # [
    #     "CNN",
    #     f"{prediction_dir}/CNN64_seq11-2-0-0_multi/{data_type}/{speaker}/{file}_dis_pred.csv",
    # ],
    # [
    #     "LSTM",
    #     f"{prediction_dir}/LSTM_u64_l1_seq0-0-0-0_multi/{data_type}/{speaker}/{file}_dis_pred.csv",
    # ],
    [
        "Transformer",
        f"{prediction_dir}/TransformerEncoder_d64_l1_h8_seq11-2-11-2_multi/direct/{data_type}/{speaker}/{file}_dis_pred.csv",
    ],
]

for plot in plots:
    df = pd.read_csv(plot[1])
    plt.plot(
        np.linspace(0,len(df)/60,len(df))[lower:upper],
        df[key][lower:upper],
        label=plot[0],
    )

plt.xlabel('time (s)')
plt.ylabel('displacement (mm)')
plt.legend()
plt.show()