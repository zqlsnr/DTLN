import onnxruntime
import soundfile as sf
import numpy as np
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

interpreter_1 = onnxruntime.InferenceSession('onnx_model/16/16k_1.onnx')
model_input_names_1 = [inp.name for inp in interpreter_1.get_inputs()]
# preallocate input
model_inputs_1 = {
            inp.name: np.zeros(
                [dim if isinstance(dim, int) else 1 for dim in inp.shape],
                dtype=np.float32)
            for inp in interpreter_1.get_inputs()}
# load models
interpreter_2 = onnxruntime.InferenceSession('onnx_model/16/16k_2.onnx')
model_input_names_2 = [inp.name for inp in interpreter_2.get_inputs()]
# preallocate input
model_inputs_2 = {
            inp.name: np.zeros(
                [dim if isinstance(dim, int) else 1 for dim in inp.shape],
                dtype=np.float32)
            for inp in interpreter_2.get_inputs()}

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

    # set block to input
    model_inputs_1[model_input_names_1[0]] = in_mag
    # run calculation
    model_outputs_1 = interpreter_1.run(None, model_inputs_1)

    mask_1 = model_outputs_1[0]
    model_inputs_1[model_input_names_1[1]] = model_outputs_1[1]

    model_inputs_2[model_input_names_2[0]] = mask_1
    # run calculation
    model_outputs_2 = interpreter_2.run(None, model_inputs_2)

    mask_2 = model_outputs_2[0]
    model_inputs_2[model_input_names_2[1]] = model_outputs_2[1]
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
sf.write('out_16_onnx.wav', out, fs)
print('Processing Time [ms]:')
print(np.mean(np.stack(time_array)) * 1000)
print(time.time() - start)
print('Processing finished.')
