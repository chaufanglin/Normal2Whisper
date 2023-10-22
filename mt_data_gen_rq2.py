import os
import numpy as np
import argparse
import soundfile as sf
import librosa
from utils import glottal_remove_gen, bandwidth_widen_gen
import threading
import time

lock = threading.Lock() 

def generate(job, data_list, output_dir, generating_mode):
    print("This is thread %s at %s " % (job, time.ctime(time.time())))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(data_list, 'r') as f1:
        content = f1.readlines()
        content = [x.strip() for x in content] 
        for line in content:
            name, filepath = line.split('  ',1)
            # filepath, _ = filepath.split(' |', 1)

            spkpath, _ = filepath.split('normal/', 1)[1].split('/s', 1)
            spkpath = os.path.join(output_dir, spkpath)
            # filename_split = os.path.basename(filepath).split('.', 1)[0].split('-')
            # spkpath = os.path.join(output_dir, filename_split[0], filename_split[1])

            # name = name.replace("n","pw")
            if os.path.exists(os.path.join(spkpath, name) + '.wav'):
                continue

            if not os.path.exists(spkpath):
                lock.acquire()
                if not os.path.exists(spkpath):
                    os.makedirs(spkpath, exist_ok=True)
                    print("make dir: %s" % spkpath)
                lock.release()

            s_n, fs = librosa.load(filepath, sr=16000, dtype=np.float64)    # resample to 16k
            # s_n, fs = sf.read(filepath)
            if generating_mode == '1':
                s_pw = glottal_remove_gen(s_n, fs)
            else:
                s_pw = bandwidth_widen_gen(s_n, fs)
            
            sf.write(os.path.join(spkpath, name) + '.wav', s_pw, fs)
    f1.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Generate pseudo whispered speech data')
    
    data_list = ['/tudelft.net/staff-bulk/ewi/insy/SpeechLab/zhaofenglin/Normal2Whisper/split/aa',
                 '/tudelft.net/staff-bulk/ewi/insy/SpeechLab/zhaofenglin/Normal2Whisper/split/ab',
                 '/tudelft.net/staff-bulk/ewi/insy/SpeechLab/zhaofenglin/Normal2Whisper/split/ac',
                 '/tudelft.net/staff-bulk/ewi/insy/SpeechLab/zhaofenglin/Normal2Whisper/split/ad',
                 '/tudelft.net/staff-bulk/ewi/insy/SpeechLab/zhaofenglin/Normal2Whisper/split/ae',
                 '/tudelft.net/staff-bulk/ewi/insy/SpeechLab/zhaofenglin/Normal2Whisper/split/af',
                 '/tudelft.net/staff-bulk/ewi/insy/SpeechLab/zhaofenglin/Normal2Whisper/split/ag',
                 '/tudelft.net/staff-bulk/ewi/insy/SpeechLab/zhaofenglin/Normal2Whisper/split/ah',]
    
    generating_mode_default = '1'
    
    parser.add_argument('--generating_mode', type = str, help = 'Generating mode: 1) glottal contribution removal; 2) formant bandwidth widen.', 
                        default = generating_mode_default)
    argv = parser.parse_args()
    generating_mode = argv.generating_mode

    output_dir = '/tudelft.net/staff-bulk/ewi/insy/SpeechLab/zhaofenglin/Normal2Whisper/wtm_pw/test' + generating_mode
    print('output dir:', output_dir)
    nj = 8
    Threads = []

    for job in range(nj):
        Threads.append(threading.Thread(target=generate, args=(job+1, data_list[job], output_dir, generating_mode)))

    for job in range(nj):
        Threads[job].start()

    for t in Threads:
        t.join()
    print("All threads are finished!")


    