# Normal2Whisper
This is an implementation of our pseudo-whispered speech conversion method in the paper Improving Whispered Speech Recognition Performance using Pseudo-whispered based Data Augmentation (to be appear in ASRU 2023).

<img src="Pseudo-whisper.png" width="60%">

## Dependencies
* Python 3.8 
* Numpy
* soundfile
* librosa
* [PyWorld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder)

## Functions
1. `utils.py`

    This script has all the essential functions used in our proposed method.

2. `data_gen.py`

    This script is used to convert normal speech into pseudo-whispered speech. 

3. `rq2_gen.py`

    This script is used to convert normal speech into: 
    1) normal speech without glottal contributions; 
    2) normal speech with widened formant bandwidth and shifted formant frequencies. 

## Usage
**1. Convert normal speech into pseudo-whispered speech from your dataset:**

```Bash
python data_gen.py --data_list './list_example(PATH TO THE LIST OF SOURCE TRAINING DATA)' --output_dir './data/training/wTIMIT/PW(PATH TO OUTPUT PW DIRECTORY)' 
```

**2. Convert normal speech into 1) normal speech without glottal contributions:**

```Bash
python rq2_gen.py --data_list './list_example(PATH TO THE LIST OF SOURCE TRAINING DATA)' --output_dir './data/training/wTIMIT/s1(PATH TO OUTPUT DIRECTORY)' --generating_mode '1'
```

**3. Convert normal speech into 2) normal speech with widened formant bandwidth and shifted formant frequencies:**

```Bash
python rq2_gen.py --data_list './list_example(PATH TO THE LIST OF SOURCE TRAINING DATA)' --output_dir './data/training/wTIMIT/s2(PATH TO OUTPUT DIRECTORY)' --generating_mode '2'
```

**Note:** you can check `./list_example` to see an example of the input data list. 
