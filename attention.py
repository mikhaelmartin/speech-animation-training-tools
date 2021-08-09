import numpy as np
import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = np.float32(d_model)

        self.angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
        )

        # apply sin to even indices in the array; 2i
        self.angle_rads[:, 0::2] = np.sin(self.angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        self.angle_rads[:, 1::2] = np.cos(self.angle_rads[:, 1::2])

        self.pos_encoder = tf.cast(self.angle_rads[np.newaxis, ...], tf.float32)

    def get_angles(self, pos, i):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(self.d_model))
        return pos * angle_rates

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoder[:, :seq_len, :]


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model, activation="linear")
        self.wk = tf.keras.layers.Dense(d_model, activation="linear")
        self.wv = tf.keras.layers.Dense(d_model, activation="linear")

        self.dense = tf.keras.layers.Dense(d_model, activation="linear")

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, q, k, v):
        """Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead)
        but it must be broadcastable for addition.

        Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
                to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
        output, attention_weights
        """

        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # # add the mask to the scaled tensor.
        # if mask is not None:
        #   scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(
            scaled_attention_logits, axis=-1
        )  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v)

        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_model)
        )  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class MultiHeadAttentionSubLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttentionSubLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, v, k, q):
        attn_output, _ = self.mha(v, k, q)  # (batch_size, input_seq_len, d_model)
        return self.layernorm(q + attn_output)  # (batch_size, input_seq_len, d_model)


class FeedForwardNetworkSubLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, dff):
        super(FeedForwardNetworkSubLayer, self).__init__()
        self.ffn = self.point_wise_feed_forward_network(d_model, dff)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def point_wise_feed_forward_network(self, d_model, dff):
        return tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    dff, activation="relu"
                ),  # (batch_size, seq_len, dff)
                tf.keras.layers.Dense(
                    d_model, activation="linear"
                ),  # (batch_size, seq_len, d_model)
            ]
        )

    def call(self, x):
        # ffn_output = self.ffn(x)  # (batch_size, input_seq_len, d_model)
        return self.layernorm(x + self.ffn(x))  # (batch_size, input_seq_len, d_model)


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttentionSubLayer(d_model, num_heads)
        self.ffn = FeedForwardNetworkSubLayer(d_model, dff)

    def call(self, x):
        return self.ffn(self.mha(x, x, x))


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff):
        super(DecoderLayer, self).__init__()

        # self attention
        self.mha1 = MultiHeadAttentionSubLayer(d_model, num_heads)
        # encoder-decoder attention
        self.mha2 = MultiHeadAttentionSubLayer(d_model, num_heads)
        self.ffn = FeedForwardNetworkSubLayer(d_model, dff)

    def call(self, x, enc_output):
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        return self.ffn(
            self.mha2(enc_output, enc_output, self.mha1(x, x, x))
        )  # (batch_size, target_seq_len, d_model)


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        maximum_position_encoding,
    ):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.input_layer = tf.keras.layers.Dense(d_model, activation="tanh")
        self.pos_encoding = PositionalEncoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff) for _ in range(num_layers)
        ]

    def call(self, x):
        x = self.input_layer(x)
        x = self.pos_encoding(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        maximum_position_encoding,
    ):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.input_layer = tf.keras.layers.Dense(d_model, activation="tanh")
        self.pos_encoding = PositionalEncoding(maximum_position_encoding, d_model)

        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff) for _ in range(num_layers)
        ]

    def call(self, x, enc_output):
        x = self.input_layer(x)
        x = self.pos_encoding(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output)

        # x.shape == (batch_size, target_seq_len, d_model)
        return x


class Transformer(tf.keras.Model):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        pe_input,
        pe_target,
        target_output_size,
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, pe_input)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, pe_target)

        self.final_layer = tf.keras.layers.Dense(
            target_output_size, activation="linear"
        )

    def call(self, enc_inp, dec_inp):
        return self.final_layer(self.decoder(dec_inp, self.encoder(enc_inp)))


# X_feature_size = 39
# Y_feature_size = 107
# x_num_pre = 11
# x_num_post = 3
# y_num_pre =11
# y_num_post = 3
# # x_pad_value= None
# # y_pad_value= None
# x_pad_value= np.array([-300]+[0.0]*38)
# y_pad_value= np.zeros((107,), dtype=float)
# x_shift = 0
# y_shift = 0
# x_seq_len = x_num_pre + x_num_post + 1
# y_seq_len = y_num_pre + y_num_post + 1
# batch_size = 64


# d_model = 64
# num_heads = 8
# dff = 256
# num_layers = 1

# enc_inp = tf.keras.Input((x_seq_len,X_feature_size), dtype='float32')
# dec_inp = tf.keras.Input((y_seq_len,Y_feature_size), dtype='float32')
# output =  Transformer(
#             num_layers=num_layers,
#             d_model=d_model,
#             num_heads=num_heads,
#             dff=dff,
#             pe_input=x_seq_len,
#             pe_target=y_seq_len,
#             target_output_size=Y_feature_size) (enc_inp)
# model = tf.keras.models.Model(inputs=enc_inp,outputs=output)
# print(model.summary())
# model.compile()

# y = model(np.zeros((100,15,39)))

# # y = model.predict(x=np.zeros((100,15,39)))

# print(y)
