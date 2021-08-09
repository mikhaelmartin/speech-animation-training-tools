import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


key = "Y_57"

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
        f"{prediction_dir}/TransformerEncoder_d64_l1_h8_seq11-2-11-2_multi/smoothing/{data_type}/{speaker}/{file}_dis_pred.csv",
    ],
]
# t = np.arange(256)
# f = np.fft.fftfreq(t.shape[-1])

Fs = 60
# f = np.fft.fftfreq()
for plot in plots:
    df = pd.read_csv(plot[1])
    X = np.fft.fft(df[key])
    X_mag = np.abs(X)/len(X)
    f = np.linspace(0,Fs,len(X_mag))
    
    plt.plot(
        f[1:int(len(f)/2)],        
        X_mag[1:int(len(f)/2)],
        label=plot[0],
    )

plt.xlabel('frequency (Hz)')
# plt.ylabel('magnitude (mm)')
plt.legend()
plt.show()