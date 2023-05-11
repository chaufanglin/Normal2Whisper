mkdir wtm_pw_16k
mkdir wtm_pw_16k/train_normal
cd wtm_pw_16k/train_normal
mkdir US
mkdir SG
sgspk="000  001  002  003  004  005  006  007  008  009  010  011  012  013  014  015  016  017  018  019"
for spk in ${sgspk};do
    echo $spk
    mkdir ./SG/$spk
done