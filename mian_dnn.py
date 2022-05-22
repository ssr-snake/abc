import h5py
import numpy as np
import os
import _pickle
import random
from keras import layers
import tensorflow as tf



_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0
n_concat = 7
batch_size = 80
iter = 0
fs = 16000
tr_hdf5_path = os.path.join('workspace', "features")
tr_hdf5_names = [na for na in os.listdir(tr_hdf5_path) if na.lower().endswith(".h5")]
random.shuffle(tr_hdf5_names)
total_filenum = len(tr_hdf5_names)
train_filenum = total_filenum
train_f_names = tr_hdf5_names

def HertzToMel(frequencies_hertz):
  return _MEL_HIGH_FREQUENCY_Q * np.log(1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))

def MelToHertz(frequencies_mel):
    """Convert frequencies to mel scale using HTK formula.

    Args:
      frequencies_hertz: Scalar or np.array of frequencies in hertz.

    Returns:
      Object of same size as frequencies_hertz containing corresponding values
      on the mel scale.
    """

    return _MEL_BREAK_FREQUENCY_HERTZ * (np.exp(frequencies_mel / _MEL_HIGH_FREQUENCY_Q) - 1)

def MelToSpectrogramMatrix(num_mel_bins=20, num_spectrogram_bins=129, audio_sample_rate=8000,
                           lower_edge_hertz=125.0, upper_edge_hertz=3800.0):

    nyquist_hertz = audio_sample_rate / 2.
    if lower_edge_hertz < 0.0:
        raise ValueError("lower_edge_hertz %.1f must be >= 0" % lower_edge_hertz)
    if lower_edge_hertz >= upper_edge_hertz:
        raise ValueError("lower_edge_hertz %.1f >= upper_edge_hertz %.1f" %
                         (lower_edge_hertz, upper_edge_hertz))
    if upper_edge_hertz > nyquist_hertz:
        raise ValueError("upper_edge_hertz %.1f is greater than Nyquist %.1f" %
                         (upper_edge_hertz, nyquist_hertz))
    spectrogram_bins_hertz = np.linspace(0.0, nyquist_hertz, num_spectrogram_bins)

    # The i'th mel band (starting from i=1) has center frequency
    # band_edges_mel[i], lower edge band_edges_mel[i-1], and higher edge
    # band_edges_mel[i+1].  Thus, we need num_mel_bins + 2 values in
    # the band_edges_mel arrays.
    band_edges_mel = np.linspace(HertzToMel(lower_edge_hertz),
                                 HertzToMel(upper_edge_hertz), num_mel_bins + 2)
    band_edges_hertz = MelToHertz(band_edges_mel)
    hertz_weight_matrix = np.zeros(shape=(num_spectrogram_bins, num_mel_bins), dtype=np.float32)
    mel_center_hertz = band_edges_hertz[1:num_mel_bins + 1]
    for row in range(num_spectrogram_bins):
        hertz = spectrogram_bins_hertz[row]
        for col in range(num_mel_bins):
            upper_hertz = mel_center_hertz[col]
            if col == 0:
                lower_hertz = 0.0
            else:
                lower_hertz = mel_center_hertz[col - 1]
            if hertz >= lower_hertz and hertz < upper_hertz:
                band_gap = abs(upper_hertz - lower_hertz)
                lower_gap = abs(hertz - lower_hertz)
                upper_gap = abs(hertz - upper_hertz)

                if col == 0:
                    hertz_weight_matrix[row][col] = lower_gap / band_gap
                else:
                    hertz_weight_matrix[row][col - 1] = upper_gap / band_gap
                    hertz_weight_matrix[row][col] = lower_gap / band_gap
            elif col == (num_mel_bins - 1) and hertz >= upper_hertz:
                band_gap = abs(nyquist_hertz - upper_hertz)
                lower_gap = abs(hertz - upper_hertz)
                hertz_weight_matrix[row][col] = (band_gap - lower_gap) / band_gap
    # hertz_weight_matrix = np.transpose(hertz_weight_matrix)
    # print("hertz_weight_matrix:")
    # print(hertz_weight_matrix)

    return hertz_weight_matrix

def SpectrogramToMelMatrix(num_mel_bins=32, num_spectrogram_bins=129,audio_sample_rate=16000,
                           lower_edge_hertz=0.0,upper_edge_hertz=8000.0):
    nyquist_hertz = audio_sample_rate / 2.
    if lower_edge_hertz < 0.0:
        raise ValueError("lower_edge_hertz %.1f must be >= 0" % lower_edge_hertz)
    if lower_edge_hertz >= upper_edge_hertz:
        raise ValueError("lower_edge_hertz %.1f >= upper_edge_hertz %.1f" %
                     (lower_edge_hertz, upper_edge_hertz))
    if upper_edge_hertz > nyquist_hertz:
        raise ValueError("upper_edge_hertz %.1f is greater than Nyquist %.1f" %
                     (upper_edge_hertz, nyquist_hertz))
    spectrogram_bins_hertz = np.linspace(0.0, nyquist_hertz, num_spectrogram_bins)
    spectrogram_bins_mel = HertzToMel(spectrogram_bins_hertz)
    band_edges_mel = np.linspace(HertzToMel(lower_edge_hertz),
                               HertzToMel(upper_edge_hertz), num_mel_bins + 2)
    mel_weights_matrix = np.empty((num_spectrogram_bins, num_mel_bins))
    for i in range(num_mel_bins):
        lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[i:i + 3]
        lower_slope = ((spectrogram_bins_mel - lower_edge_mel) /
                   (center_mel - lower_edge_mel))
        upper_slope = ((upper_edge_mel - spectrogram_bins_mel) /
                   (upper_edge_mel - center_mel))
        mel_weights_matrix[:, i] = np.maximum(0.0, np.minimum(lower_slope,
                                                          upper_slope))
    mel_weights_matrix[0, :] = 0.0
    return mel_weights_matrix

def load_hdf5(hdf5_path):
    with h5py.File(hdf5_path, 'r') as hf:
        x = hf.get('x')
        y = hf.get('y')
        x = np.array(x)
        y = np.array(y)
    return x, y

def calculate_loss(y_irm, pred, H):
    loss = pred - np.dot(y_irm, H)
    return loss

class DataGenerator(object):
    def __init__(self, batch_size, type, te_max_iter=None):
        assert type in ['train', 'test']
        self._batch_size_ = batch_size
        self._type_ = type
        self._te_max_iter_ = te_max_iter

    def generate(self, xs, ys):
        x = xs[0]
        y = ys[0]
        batch_size = self._batch_size_
        n_samples = len(x)
        index = np.arange(n_samples)
        np.random.shuffle(index)

        iter = 0
        epoch = 0
        pointer = 0
        while True:
            if (self._type_ == 'test') and (self._te_max_iter_ is not None):
                if iter == self._te_max_iter_:
                    break
            iter += 1
            if pointer >= n_samples:
                epoch += 1
                if (self._type_ == 'test') and (epoch == 1):
                    break
                pointer = 0
                np.random.shuffle(index)

            batch_idx = index[pointer: min(pointer + batch_size, n_samples)]
            pointer += batch_size
            yield x[batch_idx], y[batch_idx]

'''model = Sequential()
#model.add(layers.BatchNormalization(input_shape=(7, 40)))
model.add(layers.Reshape((9, 64, 1), input_shape=(9, 64)))
model.add(layers.Conv2D(500, strides=(1, 2), kernel_size=(2, 2)))
model.add(layers.Conv2D(1, strides=(1, 2), kernel_size=(2, 2)))
#model.add(layers.GRU(20, activation='sigmoid', recurrent_activation='tanh'))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='tanh'))
model.add(layers.Dense(5, activation='sigmoid'))

main_input = layers.Input(shape=(32, 32), name='main_input')
x1 = layers.Reshape((32, 32, 1))(main_input)
x2 = layers.Conv2D(8, (2, 2), strides=(2, 2), activation='sigmoid')(x1)
x3 = layers.MaxPool2D((2, 2))(x2)
x4 = layers.Conv2D(32, (2, 2), strides=(2, 2), activation='sigmoid')(x3)
x5 = layers.MaxPool2D((2, 2))(x4)
x6 = layers.Conv2D(128, (2, 2), strides=(2, 2), activation='sigmoid')(x5)
x7 = layers.Flatten()(x6)
dense_1 = layers.Dense(32, activation='sigmoid')(x7)
main_output = layers.Dense(8, activation='sigmoid', name='denoise_output')(dense_1)
model = tf.keras.Model(inputs=main_input, outputs=main_output)'''

# H0 = SpectrogramToMelMatrix(num_mel_bins=40, num_spectrogram_bins=257, audio_sample_rate=16000,
#                              lower_edge_hertz=0.0, upper_edge_hertz=8000.0)
# H0 = H0.T

main_input = layers.Input(shape=(7, 40), name='main_input')
gru_1 = layers.GRU(32, activation='sigmoid', recurrent_activation='tanh', reset_after=False, return_sequences=False)(main_input)
tmp = layers.Reshape((1, 32))(gru_1)
gru_2 = layers.GRU(16, activation='sigmoid', recurrent_activation='tanh', reset_after=False)(tmp)
input = layers.concatenate([gru_1, gru_2])
input = layers.Reshape((1, 48))(input)
gru_3 = layers.GRU(16, activation='sigmoid', recurrent_activation='tanh', reset_after=False)(input)
denoise_gru = layers.concatenate([gru_1, gru_2, gru_3])
tmp = layers.Reshape((1, 64))(denoise_gru)
gru_4 = layers.GRU(32, reset_after=False)(tmp)
flatten_1 = layers.Flatten()(gru_4)
#dense_1 = layers.Dense(40, activation='sigmoid')(flatten_1)
main_output = layers.Dense(257, activation='sigmoid', name='denoise_output')(flatten_1)
model = tf.keras.Model(inputs=main_input, outputs=main_output)

model.summary()
model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
tr_gen = DataGenerator(batch_size=batch_size, type='train')
loss_sum = 0


'''
for m in range(100):
    for i in range(train_filenum):
        (tr_x, tr_y) = load_hdf5(os.path.join(tr_hdf5_path, train_f_names[i]))
        iter = iter + 1
        ite = 0
        epic = int(len(tr_y)/batch_size)
        for batch_x, batch_y in tr_gen.generate(xs=[tr_x], ys=[tr_y]):
            loss = model.train_on_batch(batch_x, batch_y)
            loss_sum = loss_sum + loss
            ite += 1
            iter += 1
            if iter % 10 == 0:
                model_path = os.path.join('workspace', 'models', "md_%diters.h5" % iter)
                model.save(model_path)
                print("Saved model to %s" % model_path)
                print(loss_sum / 10)
                loss_sum = 0
            if ite > epic:
                break

            if (loss2 < best_loss) or (np.abs(best_loss-loss2) < 1e-4):
                if best_loss < loss2:
                    best_loss = loss2
                model_path = os.path.join('workspace/models', "md_best%d.h5" % iter)
                model.save(model_path)'''



scaler = _pickle.load(open('workspace/scalers/feature_standard_scaler.p', 'rb'))
mean = np.array(scaler.mean_)
var = scaler.var_

f = open('loss.csv', 'w')
for m in range(100):
    for i in range(train_filenum):
        (tr_x, tr_y) = load_hdf5(os.path.join(tr_hdf5_path, train_f_names[i]))
        tr_x_shape = np.shape(tr_x)
        tr_y_shape = np.shape(tr_y)
        '''for i1 in range(tr_x_shape[0]):
            for i2 in range(tr_x_shape[1]):
                tr_x[i1, i2, :] = (tr_x[i1, i2, :] - mean) / var
        for i1 in range(tr_y_shape[0]):
            for i2 in range(tr_y_shape[1]):
                if tr_y[i1, i2] > 1:
                    tr_y[i1, i2] = 1'''
        #ite = 0
        #epic = int(len(tr_y)/batch_size)
        for batch_x, batch_y in tr_gen.generate(xs=[tr_x], ys=[tr_y]):
            loss = model.train_on_batch(batch_x, batch_y)
            loss_sum = loss_sum + loss
            #ite += 1
            iter += 1
            if iter % 5000 == 0:
                model_path = os.path.join('workspace', 'models', "md_%diters.h5" % iter)
                model.save(model_path)
                print("Saved model to %s" % model_path)
                print(loss_sum/5000)
                f.write("%d\t%f\n" % (iter, loss_sum/5000))
                loss_sum = 0
            #if ite > epic:
             #   break
f.close()



