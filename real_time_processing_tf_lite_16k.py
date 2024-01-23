import soundfile as sf
import numpy as np
import tflite_runtime.interpreter as tflite
import time


def overlapadd(frame, hop=128):
    N_frame, L_frame = frame.shape
    length = L_frame + (N_frame - 1) * hop
    output = np.zeros(length).astype('float32')
    for i in range(N_frame):
        output[hop * i: hop * i + L_frame] += frame[i]

    return output


start = time.time()
block_len = 512
block_shift = 128

# load models
interpreter_1 = tflite.Interpreter(model_path='model/16/16k_1.tflite')
interpreter_1.allocate_tensors()
interpreter_2 = tflite.Interpreter(model_path='model/16/16k_2.tflite')
interpreter_2.allocate_tensors()

# Get input and output tensors.
input_details_1 = interpreter_1.get_input_details()
output_details_1 = interpreter_1.get_output_details()

input_details_2 = interpreter_2.get_input_details()
output_details_2 = interpreter_2.get_output_details()
# create states for the lstms
states_1 = np.zeros(input_details_1[1]['shape']).astype('float32')
states_2 = np.zeros(input_details_2[1]['shape']).astype('float32')
# load audio file at 16k fs (please change)
audio, fs = sf.read('test/59f84d_1.wav', dtype='float32')

# preallocate output audio
out_file = np.zeros((len(audio)))
# calculate number of blocks
num_blocks = (audio.shape[0] - (block_len - block_shift)) // block_shift
time_array = []
win = np.sin(np.arange(.5, block_len - .5 + 1) / block_len * np.pi).astype('float32')
out_array = []
# iterate over the number of blcoks
for idx in range(num_blocks):
    start_time = time.time()
    audio_buffer = audio[idx * block_shift:(idx * block_shift) + block_len]
    # calculate fft of input block
    in_buffer = audio_buffer * win
    in_block_fft = np.fft.rfft(in_buffer)
    in_mag = np.abs(in_block_fft)
    in_phase = np.angle(in_block_fft)
    # reshape magnitude to input dimensions
    in_mag = np.reshape(in_mag, (1, 1, -1)).astype('float32')
    # set tensors to the first model
    interpreter_1.set_tensor(input_details_1[1]['index'], states_1)
    interpreter_1.set_tensor(input_details_1[0]['index'], in_mag)
    # run calculation
    interpreter_1.invoke()
    # get the output of the first block
    mask_1 = interpreter_1.get_tensor(output_details_1[0]['index'])
    states_1 = interpreter_1.get_tensor(output_details_1[1]['index'])

    # set tensors to the second block
    interpreter_2.set_tensor(input_details_2[1]['index'], states_2)
    interpreter_2.set_tensor(input_details_2[0]['index'], mask_1)
    # run calculation
    interpreter_2.invoke()
    # get output tensors
    mask_2 = interpreter_2.get_tensor(output_details_2[0]['index'])
    states_2 = interpreter_2.get_tensor(output_details_2[1]['index'])

    # calculate the ifft
    estimated_complex = in_mag * mask_2 * np.exp(1j * in_phase)
    estimated_block = np.fft.irfft(estimated_complex).astype('float32')
    estimated_block = estimated_block * win
    out_array.append(estimated_block)
    out_file[block_shift * idx: block_shift * idx + block_len] += np.squeeze(estimated_block)
    time_array.append(time.time() - start_time)

output = np.squeeze(np.concatenate(out_array, axis=1))
out = overlapadd(output)
# write to .wav file
sf.write('out_test/out_59f84d_1.wav', out, fs)
print('Processing Time [ms]:')
print(np.mean(np.stack(time_array)) * 1000)
print(time.time() - start)
print('Processing finished.')
