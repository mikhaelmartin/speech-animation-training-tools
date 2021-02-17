import matplotlib.pyplot as plt
import pandas as pd
import IPython


rnn_history_df = pd.read_csv( "../saved_model/RNN64_s19_batch64_seq0-0-0-0/RNN64_s19_batch64_seq0-0-0-0-history.csv")
gru_history_df = pd.read_csv( "../saved_model/GRU64_s19_batch64_seq0-0-0-0/GRU64_s19_batch64_seq0-0-0-0-history.csv")
lstm_history_df = pd.read_csv( "../saved_model/LSTM64_s19_batch64_seq0-0-0-0/LSTM64_s19_batch64_seq0-0-0-0-history.csv")
attention_history_df = pd.read_csv( "../saved_model/Attention64_s19_mha1_numheads8_ffn0_batch64_seq12-12-0-0/Attention64_s19_mha1_numheads8_ffn0_batch64_seq12-12-0-0-history.csv")
attention_ffn_history_df = pd.read_csv( "../saved_model/Attention64_s19_mha1_numheads8_ffn1_dff256_batch64_seq12-12-0-0/Attention64_s19_mha1_numheads8_ffn1_dff256_batch64_seq12-12-0-0-history.csv")



# ax = plt.subplot(111)
plt.plot(rnn_history_df["timestamp"][:155][::2],rnn_history_df["val_loss"][:155][::2],label="RNN")
plt.plot(gru_history_df["timestamp"][:140][::2],gru_history_df["val_loss"][:140][::2],label="GRU")
plt.plot(lstm_history_df["timestamp"][:140][::2],lstm_history_df["val_loss"][:140][::2],label="LSTM")
plt.plot(attention_history_df["timestamp"][::2],attention_history_df["val_loss"][::2],label="Multi Head Attention")
plt.plot(attention_ffn_history_df["timestamp"][::2],attention_ffn_history_df["val_loss"][::2],label="Multi Head Attention + FNN")
plt.ylabel('Validation Loss')

plt.yscale('log')
# plt.xscale('log')
# plt.title("Validation Loss: Mean Squared Error")
plt.xlabel("Time[sec]")
plt.grid()
plt.legend()
plt.show()



