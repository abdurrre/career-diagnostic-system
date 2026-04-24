import tensorflow as tf
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Dropout, BatchNormalization

class NERModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units, num_classes, **kwargs):
        super(NERModel, self).__init__(**kwargs)
        self.embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True, name="ner_embed")
        self.dropout = Dropout(0.3)
        self.bilstm = Bidirectional(LSTM(units=rnn_units, return_sequences=True), name="ner_bilstm")
        self.classifier = TimeDistributed(Dense(units=num_classes, activation='softmax'), name="ner_out")

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        if training: x = self.dropout(x, training=training)
        x = self.bilstm(x)
        return self.classifier(x)

class ScoringModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(ScoringModel, self).__init__(**kwargs)
        self.dense1 = Dense(256, activation='relu')
        self.bn1 = BatchNormalization()
        self.dense2 = Dense(64, activation='relu')
        self.out_layer = Dense(1, activation='sigmoid', name="score_out")

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)
        x = self.dense2(x)
        return self.out_layer(x)

class GapModel(tf.keras.Model):
    def __init__(self, num_professions, num_skills, embedding_dim=32, **kwargs):
        super(GapModel, self).__init__(**kwargs)
        self.prof_embedding = Embedding(num_professions, embedding_dim)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = Dense(256, activation='relu')
        self.out_layer = Dense(num_skills, activation='sigmoid', name="gap_out")

    def call(self, inputs, training=False):
        x = self.prof_embedding(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.out_layer(x)
