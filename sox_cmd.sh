#!/bin/bash  
for x in ./Normal2Whisper/wtm_pw/*.wav;do
    echo ${x}
    b=${x##*/}  
    sox $b -r 16000 tmp-$b  
    rm -rf $b  
    mv tmp-$b $b  
done