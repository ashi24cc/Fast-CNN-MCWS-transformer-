def dictionary(chunk_size):
    dataframe = pd.read_csv("/content/gdrive/My Drive/Transformer_positional_embedding/data2017/mf/trainData.csv", header=None)
    dataset = dataframe.values
    del dataframe

    seq_dataset = dataset[:,0]
    print('Creating Dictionary:')
    dict = {}
    j = 0
    for row in seq_dataset:
        for i in range(len(row) - chunk_size + 1):
            key = row[i:i + chunk_size]
            if key not in dict:
                dict[key] = j
                j = j + 1
    del dataset, seq_dataset
    return(dict)

def nGram(dataset, chunk_size, dictI):
    dict1 = list()
    for j, row in enumerate(dataset):
        string = row
        dict2 = list()
        for i in range(len(string) - chunk_size + 1):
            try:
                dict2.append(dictI[string[i:i + chunk_size]])
            except:
                None
        dict1.append(dict2)
    return(dict1)

# CREATING DICTIONARY
chunkSize = 4
dict_Prop = dictionary(chunkSize)
max_seq_len = segmentSize - chunkSize + 1

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, seq_len, embed_dim, w_size, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.w_size = w_size
        if embed_dim % num_heads != 0:
            raise ValueError(f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}")
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def create_window_mask(self, seq_len, w_size):
        seq_len = 48
        mat = np.ones((seq_len, seq_len))
        for index in range(seq_len):
            for j in range(max(0, index - w_size), min(index + w_size + 1, seq_len)):
                mat[index][j] = 0
        tensor = tf.convert_to_tensor(mat)
        return tf.cast(tensor, tf.bool)

    def attention(self, query, key, value, mask):
        score = tf.matmul(query, key, transpose_b=True)                                  # Calculate attention.
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)

        # Implement Masking.
        if self.w_size is not None and mask is not None:
            win_mask = self.create_window_mask(self.seq_len, self.w_size)                # Compute window mask.
            int_mask = tf.math.logical_or(tf.cast(mask[0], tf.bool), win_mask)           # Combine pad mask and window mask.
            int_mask = tf.math.logical_or(tf.cast(mask[1], tf.bool), tf.cast(int_mask, tf.bool))
            final_mask = tf.cast(int_mask, tf.float32)
            scaled_score += (final_mask * -1e5)
        elif mask is not None:                                                            # add the mask to the scaled tensor.
            final_mask = mask[0]
            scaled_score += (final_mask * -1e5)                                           # mask: Float tensor (..., seq_len_q, seq_len_k).

        weights = tf.nn.softmax(scaled_score, axis=-1)
        max_wt = tf.reduce_max(weights, axis = -1)
        scaled_weights = tf.divide(weights, max_wt[:,:,:,tf.newaxis])                     # Scaled SOFTMAX
        reverse_final_mask = 1.0 - final_mask
        scaled_weights = tf.multiply(scaled_weights, reverse_final_mask) + 0.001          # Re-apply mask
        output = tf.matmul(scaled_weights, value)
        return output, scaled_weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, mask):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)                                        # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)                                            # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)                                        # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(query, batch_size)                          # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(key, batch_size)                              # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(value, batch_size)                          # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value, mask)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])                             # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))         # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(concat_attention)                                      # (batch_size, seq_len, embed_dim)
        return output, weights

class TransformerBlock(layers.Layer):
    def __init__(self, seq_len, embed_dim, num_heads, ff_dim, w_size, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.w_size = w_size
        self.att = MultiHeadSelfAttention(seq_len, embed_dim, w_size, num_heads)                              # Sub-layer 1
        self.ffn = keras.Sequential([layers.Dense(ff_dim, kernel_initializer='normal', activation="relu"),    # Sub-layer 2
                                     layers.Dense(embed_dim, kernel_initializer='normal'),])                  # Two linear transformations with ReLU activation in between.
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate=0.2)
        self.dropout2 = layers.Dropout(rate=0.2)

    def call(self, inputs, mask, training = True):                                                                    # Main transformer block
        attn_output, attn_wt = self.att(inputs, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output), attn_wt

    def get_config(self):
        config = super().get_config().copy()
        config.update({'embed_dim': self.embed_dim, 'num_heads': self.num_heads, 'ff_dim': self.ff_dim,
                       'seq_len': self.seq_len, 'w_size': self.w_size})
        return config

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, emded_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.emded_dim = emded_dim
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=emded_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=emded_dim)
        self.dropout1 = layers.Dropout(rate=0.3)
        self.dropout2 = layers.Dropout(rate=0.3)

    def create_padding_mask(self, seq, batch_size):
        index = [[i] for i in range(0, max_seq_len - 1, 2)]                                        # Captures the alternate indices to implement sequence shorting.
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)                                           # Add extra dimensions to add the padding to the attention logits.
        seq = tf.reshape(tf.gather(seq, indices = index, axis = -1), [batch_size, len(index)])     # Helps extract the alternate values from the seq
        return seq[:, tf.newaxis, tf.newaxis, :], seq[:, tf.newaxis, :, tf.newaxis]
        # return (batch_size, 1, 1, seq_len), (batch_size, 1, seq_len, 1)

    def call(self, x, training = True):
        maxlen = tf.shape(x)[-1]
        batch_size = tf.shape(x)[0]
        padding_mask = self.create_padding_mask(x, batch_size)

        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        positions = self.dropout1(positions, training=training)
        x = self.token_emb(x)
        x = self.dropout2(x, training=training)
        return x + positions, padding_mask

    def get_config(self):
        config = super().get_config().copy()
        config.update({'maxlen': self.maxlen, 'vocab_size': self.vocab_size, 'emded_dim': self.emded_dim,})
        return config

def create_rec_model1(top_words, seq_len, o_dim):
    embedding_vecor_length = 64          # Embedding size for each token
    num_heads = 4                        # Number of attention heads
    ff_dim = 128                         # Hidden layer size in feed forward network inside transformer
    n_filters = 128                      # Number of filters with CNNs

    inputs = layers.Input(shape=(seq_len,))
    embedding_layer = TokenAndPositionEmbedding(seq_len, top_words, embedding_vecor_length)
    x, pad_mask = embedding_layer(inputs)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(n_filters, 3, padding = 'same')(x)
    x = layers.AveragePooling1D(pool_size = 2, strides = 2)(x)
    # pool_dim = int(tf.shape(x)[-2], dtype = np.int32)

    transformer_block1 = TransformerBlock(seq_len, n_filters, num_heads, ff_dim, 12)                # 13 - 1   ---------> 25 %
    x1, attn_score1 = transformer_block1(x, pad_mask)

    transformer_block2 = TransformerBlock(seq_len, n_filters, num_heads, ff_dim, 24)                # 25 - 1   ---------> 50 %
    x2, attn_score2 = transformer_block2(x, pad_mask)

    transformer_block3 = TransformerBlock(seq_len, n_filters, num_heads, ff_dim, seq_len)           # 50      ---------> Complete
    x3, attn_score3 = transformer_block3(x, pad_mask)

    x = layers.Concatenate(axis = -1)([x1, x2, x3])
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(o_dim, kernel_initializer='normal', activation='sigmoid')(x)

    r_model = keras.Model(inputs=[inputs], outputs=[outputs])

    adam = keras.optimizers.Adam(learning_rate=0.0008)
    r_model.compile(loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=3.0), optimizer=adam, metrics=['binary_accuracy'])
    return r_model

#CREATING N-GRAM
x_train = nGram(x_tr, chunkSize, dict_Prop)
x_validate = nGram(x_val, chunkSize, dict_Prop)
#del x_tr, x_val

# truncate and pad input sequences
x_train = sequence.pad_sequences(x_train, maxlen=max_seq_len)
x_validate = sequence.pad_sequences(x_validate, maxlen=max_seq_len)

# Create & Compile the model
model = create_rec_model1(len(dict_Prop), max_seq_len, nb_of_cls)
print(model.summary())
