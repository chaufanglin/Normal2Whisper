import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import pickle

# pw_data_dir = '/tudelft.net/staff-bulk/ewi/insy/SpeechLab/zhaofenglin/Normal2Whisper'
# pw_energy = []
# # for subset in ['train_dir', 'dev_dir', 'test_dir']:
# #     for file in os.listdir(os.path.join(pw_data_dir, subset)):
# #             filepath = os.path.join(pw_data_dir, subset, file)
# #             wav, fs = sf.read(filepath)
# #             pw_energy.append(sum(abs(wav**2)))

# w_energy = []
# with open('/tudelft.net/staff-bulk/ewi/insy/SpeechLab/zhaofenglin/Normal2Whisper/wtm_usw_test_wav.scp', 'r') as f1:
#     content = f1.readlines()
#     content = [x.strip() for x in content] 
#     for line in content:
#         name, filepath = line.split('  ',1)
#         filepath, _ = filepath.split(' |', 1)

#         wav, fs = sf.read(filepath)
#         w_energy.append(sum(abs(wav**2)))
# f1.close()

# pickle.dump(w_energy, open('wtm_usw_energy.pkl', 'wb'))

F=open(r'/tudelft.net/staff-bulk/ewi/insy/SpeechLab/zhaofenglin/Normal2Whisper/tm_energy.pkl','rb')
tm_energy=pickle.load(F)

F=open(r'/tudelft.net/staff-bulk/ewi/insy/SpeechLab/zhaofenglin/Normal2Whisper/pw_energy.pkl','rb')
pw_energy=pickle.load(F)

F=open(r'/tudelft.net/staff-bulk/ewi/insy/SpeechLab/zhaofenglin/Normal2Whisper/wtm_usw_energy.pkl','rb')
w_energy=pickle.load(F)

# plot
bins = np.linspace(0, 1000, 200)

plt.figure()
plt.subplot(311)
plt.hist(tm_energy, bins, alpha = 0.5)
plt.xlabel('Energy')
plt.ylabel('Number')
plt.title('Normal')
plt.subplot(312)
plt.hist(w_energy, bins, alpha = 0.5)
plt.xlabel('Energy')
plt.ylabel('Number')
plt.title('whisper')
plt.subplot(313)
plt.hist(pw_energy, bins, alpha = 0.5)
plt.xlabel('Energy')
plt.ylabel('Number')
plt.title('Pseudo whisper')
plt.tight_layout()
plt.savefig("/tudelft.net/staff-bulk/ewi/insy/SpeechLab/zhaofenglin/Normal2Whisper/energy.jpg")