cd wtm_pw/dev_normal

sgspk="000  001  002  003  004  005  006  007  008  009  010  011  012  013  014  015  016  017  018  019"
for spk in ${sgspk};do
    echo $spk
    cd ./SG/$spk
    echo $PWD
    rename n.wav pw.wav *n.wav
    cd ../..
done

usspk="101  102  103  104  105  106  107  108  109  110  111  112  116  117  118  119  120  121  122  123  124  125  126  127  128  129  130  131"
for spk in ${usspk};do
    echo $spk
    cd ./US/$spk
    echo $PWD
    rename n.wav pw.wav *n.wav
    cd ../..
done
