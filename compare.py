
with open('./Normal2Whisper/wtm_pw_16k/finished_sort', 'r') as f:
    already = f.readlines()
    already = [x.strip() for x in already] 
f.close()

rest = []
with open('./Normal2Whisper/wtm_tr_n_wav.scp', 'r') as f1:
    content = f1.readlines()
    content = [x.strip() for x in content] 
    for line in content:
            name, filepath = line.split('  ',1)
            if (name not in already):
                # print(name)
                rest.append(line + '\n')
f1.close()

with open('./Normal2Whisper/rest','w') as f2: 
    f2.writelines(rest) 
