import os
import numpy as np
import argparse
import soundfile as sf
import librosa
from utils import pesudo_whisper_gen

def generate(data_list, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(data_list, 'r') as f1:
        content = f1.readlines()
        content = [x.strip() for x in content] 
        for line in content:
            name, filepath = line.split('  ',1)
            # filepath, _ = filepath.split(' |', 1)

            # spkpath, _ = filepath.split('normal/', 1)[1].split('/s', 1)
            # spkpath = os.path.join(output_dir, spkpath)
            filename_split = os.path.basename(filepath).split('.', 1)[0].split('-')
            spkpath = os.path.join(output_dir, filename_split[0], filename_split[1])

            # name = name.replace("n","pw")
            if os.path.exists(os.path.join(spkpath, name) + '.wav'):
                continue

            if not os.path.exists(spkpath):
                os.makedirs(spkpath, exist_ok=True)
                print("make dir: %s" % spkpath)
            
            s_n, fs = librosa.load(filepath, sr=16000, dtype=np.float64)  # resample to 16k
            # s_n, fs = sf.read(filepath)
            s_pw = pesudo_whisper_gen(s_n, fs)
            
            sf.write(os.path.join(spkpath, name) + '.wav', s_pw, fs)
    f1.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Generate pseudo whispered speech data')

    data_list_default = '/tudelft.net/staff-bulk/ewi/insy/SpeechLab/zhaofenglin/Normal2Whisper/1'
    # output_dir_default = '/tudelft.net/staff-bulk/ewi/insy/SpeechLab/zhaofenglin/Normal2Whisper/output_dir'
    output_dir_default = '/tudelft.net/staff-bulk/ewi/insy/SpeechLab/zhaofenglin/LibriSpeech/train-clean-100-pw'

    parser.add_argument('--data_list', type = str, help = 'List for the normal speech directories.', default = data_list_default)
    parser.add_argument('--output_dir', type = str, help = 'Directory for the output pesudo whispered speech.', default = output_dir_default)

    argv = parser.parse_args()

    data_list = argv.data_list
    output_dir = argv.output_dir

    generate(data_list, output_dir)