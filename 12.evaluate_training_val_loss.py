from numpy.core.function_base import linspace
import pandas as pd
import matplotlib.pyplot as plt

model_dir = "../saved_model"

files = [
    [
        "MLP",
        f"{model_dir}/MLP_u64_l3_seq11-2-11-2_multi/MLP_u64_l3_seq11-2-11-2_multi-history.csv",
    ],
    [
        "LSTM",
        f"{model_dir}/LSTM_u64_l1_seq0-0-0-0_multi/LSTM_u64_l1_seq0-0-0-0_multi-history.csv",
    ],
    [
        "CNN",
        f"{model_dir}/CNN64_seq11-2-0-0_multi/CNN64_seq11-2-0-0_multi-history.csv",
    ],
    [
        "Transformer",
        f"{model_dir}/TransformerEncoder_d64_l1_h8_seq11-2-11-2_multi/TransformerEncoder_d64_l1_h8_seq11-2-11-2_multi-history.csv",
    ],
]

plt.figure()
plt.title("Validation Loss: Mean Squared Error")
plt.xlabel("Epoch")
plt.yscale("log", base=10)
plt.xscale('log',base=10)
# plt.xticks([1, 10, 100, 200], [1, 10, 100, 200])
plt.yticks([4e-3, 5e-3, 1e-2])
# plt.ylim(4e-3, 1.2e-2)
# plt.grid(True)
for file in files:
    val_loss = pd.read_csv(file[1])["val_loss"].tolist()
    plt.plot(range(1, len(val_loss) + 1), val_loss, label=file[0])
plt.legend()

plt.show()
