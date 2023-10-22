import os
import argparse
import soundfile as sf
from utils import pesudo_whisper_gen
import pickle


def generate(data_list, output_dir):
    pw_energy = []
    tm_energy = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # train_dir = os.path.join(data_dir, 'TRAIN')
    # test_dir = os.path.join(data_dir, 'TEST')
    # for file in os.listdir(data_dir):
    #     filepath = os.path.join(data_dir, file)

    with open(data_list, 'r') as f1:
        content = f1.readlines()
        content = [x.strip() for x in content] 
        for line in content:
            name, filepath = line.split('  ',1)
            name = name.replace("n", "pw");
            filepath, _ = filepath.split(' |', 1)

            s_n, fs = sf.read(filepath)
            s_pw = pesudo_whisper_gen(s_n, fs)
            
            sf.write(os.path.join(output_dir, name) + '.wav', s_pw, fs)

            # Energy
            tm_energy.append(sum(abs(s_n**2)))
            pw_energy.append(sum(abs(s_pw**2)))

        pickle.dump(pw_energy, open('pw_energy.pkl', 'wb'))
        pickle.dump(tm_energy, open('tm_energy.pkl', 'wb'))    


    f1.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Generate pesudo whispered speech data')

    data_list_default = '/tudelft.net/staff-bulk/ewi/insy/SpeechLab/zhaofenglin/Normal2Whisper/wtm_usn_test_wav.scp'
    # data_dir_default = "/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/TIMIT"
    # output_dir_default = '/tudelft.net/staff-bulk/ewi/insy/SpeechLab/zhaofenglin/Normal2Whisper/output_dir'
    output_dir_default = '/tudelft.net/staff-bulk/ewi/insy/SpeechLab/zhaofenglin/Normal2Whisper/wtm_pw/usn_test'

    parser.add_argument('--data_list', type = str, help = 'List for the normal speech directories.', default = data_list_default)
    # parser.add_argument('--data_dir', type = str, help = 'Directory for the normal speech.', default = data_dir_default)
    parser.add_argument('--output_dir', type = str, help = 'Directory for the output pesudo whispered speech.', default = output_dir_default)

    argv = parser.parse_args()

    data_list = argv.data_list
    # data_dir = argv.data_dir
    output_dir = argv.output_dir

    generate(data_list, output_dir)