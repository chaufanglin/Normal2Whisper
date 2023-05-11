find dev_dir/ -name "*.wav" > dev_list
cat dev_list | awk '{sub(/dev/,"/tudelft.net/staff-bulk/ewi/insy/SpeechLab/zhaofenglin/Normal2Whisper/dev")}1' > dev_list1

cat dev_list1 | awk '{split($0,a,"/");split(a[10],b,"."); print b[1],"sph2pipe -f wav ",$0,"|"}' > ./data_dev/wav.scp
cat dev_list1 | awk '{split($0,a,"/");split(a[10],b,".");split(b[1],c,"u"); split(c[1],d,"_"); print b[1], d[1]}' >./data_dev/utt2spk
cat text_dev | awk '{sub(/ /,"_pw ")}1' > ./data_dev/text

# rm -f dev_list1 dev_list