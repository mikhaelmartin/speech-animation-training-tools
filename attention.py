import numpy as np
import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()

        self.angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model,
        )

        # apply sin to even indices in the array; 2i
        self.angle_rads[:, 0::2] = np.sin(self.angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        self.angle_rads[:, 1::2] = np.cos(self.angle_rads[:, 1::2])

        self.pos_encoding = tf.cast(self.angle_rads[np.newaxis, ...], tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]


def scaled_dot_product_attention(q, k, v):
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
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v)

        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_model)
        )  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(dff, activation="tanh"),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(
                d_model, activation="linear"
            ),  # (batch_size, seq_len, d_model)
        ]
    )


class MultiHeadAttentionSubLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttentionSubLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        attn_output, _ = self.mha(x, x, x)  # (batch_size, input_seq_len, d_model)
        return self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)


class FeedForwardNetworkSubLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, dff):
        super(FeedForwardNetworkSubLayer, self).__init__()
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        ffn_output = self.ffn(x)  # (batch_size, input_seq_len, d_model)
        return self.layernorm2(x + ffn_output)  # (batch_size, input_seq_len, d_model)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff):
        super(Encoder, self).__init__()
        self.mha_sublayer = MultiHeadAttentionSubLayer(d_model, num_heads)
        self.ffn_sublayer = FeedForwardNetworkSubLayer(d_model, dff)

    def call(self, x):
        out1 = self.mha_sublayer(x)  # (batch_size, input_seq_len, d_model)
        out2 = self.ffn_sublayer(out1)  # (batch_size, input_seq_len, d_model)
        return out2
