import tensorflow as tf
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Dropout, BatchNormalization

class NERModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units, num_classes, **kwargs):
        super(NERModel, self).__init__(**kwargs)
        # Simpan parameter sebagai atribut biar bisa di-save
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.num_classes = num_classes

        self.embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True, name="ner_embed")
        self.dropout = Dropout(0.3)
        self.bilstm = Bidirectional(LSTM(units=rnn_units, return_sequences=True), name="ner_bilstm")
        self.classifier = TimeDistributed(Dense(units=num_classes, activation='softmax'), name="ner_out")

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        if training: x = self.dropout(x, training=training)
        x = self.bilstm(x)
        return self.classifier(x)

    def get_config(self):
        config = super(NERModel, self).get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "rnn_units": self.rnn_units,
            "num_classes": self.num_classes
        })
        return config

class GapModel(tf.keras.Model):
    def __init__(self, num_professions, num_skills, embedding_dim=32, dense_units=256, dropout_rate=0.0, **kwargs):
        super(GapModel, self).__init__(**kwargs)
        
        self.num_professions = num_professions
        self.num_skills = num_skills
        self.embedding_dim = embedding_dim
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate

        self.prof_embedding = Embedding(num_professions, embedding_dim, name="gap_embed")
        self.flatten = tf.keras.layers.Flatten(name="gap_flatten")
        self.dense1 = Dense(dense_units, activation='relu', name="gap_dense")
        self.dropout = Dropout(dropout_rate, name="gap_dropout")
        self.out_layer = Dense(num_skills, activation='sigmoid', name="gap_out")

        # FIX: Lakukan dummy forward pass agar semua weight/variable terinisialisasi
        # Hal ini mencegah error "received 0 variables during loading" di evaluate.py
        dummy_input = tf.zeros((1, 1), dtype=tf.int32)
        self(dummy_input, training=False)

    def call(self, inputs, training=False):
        x = self.prof_embedding(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        if training and self.dropout_rate > 0.0:
            x = self.dropout(x, training=training)
        return self.out_layer(x)

    def get_config(self):
        config = super(GapModel, self).get_config()
        config.update({
            "num_professions": self.num_professions,
            "num_skills": self.num_skills,
            "embedding_dim": self.embedding_dim,
            "dense_units": self.dense_units,
            "dropout_rate": self.dropout_rate
        })
        return config
