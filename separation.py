import glob
import os
import time
import numpy as np
import jams 
from jams import display
import scaper
import librosa
import soundfile as sf
import pandas as pd
import argparse


def create_tag(split_dir=None, generate_audio=True, target_sr=16000, out_dir_name='tag-new'):
    '''
    Create the tag dataset based on the given directory of jams files
    Params:
    ------
    split_dir : path to directory with jams files
    generate_audio : If True, generate Tag audio files as well
    '''
    
    paths = glob.glob(os.path.join(split_dir, '*.jams'))

    columns = ['file_name', 'source_file', 'start_time', 'label']
    df = pd.DataFrame(columns=columns)
    index = 0
    csv_dir = os.path.join(split_dir.replace('synthetic', out_dir_name).split('jams')[0], 'ann')
    openness, fold, split = split_dir.split('/')[-3:]

    for jamsPath in paths:
        fName = os.path.splitext(jamsPath)[0].replace('jams', 'audio') + '.wav'

        jamsFile = jams.load(jamsPath)

        if not os.path.isfile(fName):
            try:
                audioFile = scaper.generate_from_jams(jamsPath, fName)
            except:
                with open(f'/home/s/ss645/mlos/logs/{openness}.{fold}.{split}.txt', 'a') as f:
                    f.write('Scaper:'+fName+'\n')
                continue
            
        try:
            audioArray, _ = librosa.load(fName, sr = target_sr)
        except:
            with open(f'/home/s/ss645/mlos/logs/{openness}.{fold}.{split}.txt', 'a') as f:
                f.write('Librosa:'+fName+'\n')
            continue
        
        try:
            os.remove(fName)
        except:
            with open(f'/home/s/ss645/mlos/logs/{openness}.{fold}.{split}.txt', 'a') as f:
                f.write('Deletion:'+fName+'\n')

        annotations = jamsFile.annotations.search(namespace='scaper')[0].data
        eventCount = len(annotations[1:])

        labels = []
        start_times = []
        end_times = []
        for event in annotations[1:]:
            labels.append(event.value['label'])
            start_times.append(event.time)
            end_times.append(event.time + event.duration)

        for i in range(eventCount):

            fileLabel = []
            fileLabel.append(labels[i])
            startTime = start_times[i]
            endTime = end_times[i]    

            midPoint = (startTime + endTime) / 2
            startTime = midPoint - 0.5 # window start time

            if startTime < 0:
                startTime = 0
            elif (startTime + 1) >= 10: # probably not needed as latest event start time is 9s
                extra = (startTime + 1) - 10
                startTime = startTime - extra    
            endTime = startTime + 1 # window end time

            other_idx = [k for k in range(eventCount)]
            other_idx.pop(i)
            for j in other_idx:
                if ((startTime < start_times[j] < endTime)
                or (startTime < end_times[j] < endTime)
                or ((start_times[j] < startTime)
                and (end_times[j] > endTime))):
                    fileLabel.append(labels[j])

            sampleStart = int(startTime * target_sr)

            eventArray = audioArray[sampleStart: sampleStart + target_sr]

            trimfName = fName.replace('.wav', '_' + str(i+1) + '.wav').replace('synthetic', out_dir_name)

            if generate_audio:
                try:
                    sf.write(trimfName, eventArray, target_sr)
                except:
                    with open(f'/home/s/ss645/mlos/out/{openness}.{fold}.{split}.txt', 'a') as f:
                        f.write(trimfName+'\n')

            df2 = pd.DataFrame.from_dict({index: [trimfName, jamsPath.split('/')[-1], startTime, fileLabel]},
            orient='index', columns=columns)
            df = pd.concat([df, df2], ignore_index = True, axis = 0)
            index = index + 1

    df.to_csv(os.path.join(csv_dir, f'{openness}_{fold}_{split}.csv'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--jams_dir', help='Parent directory with synthetic jams files')
    parser.add_argument('--openness', help='\{high, mid\}')
    parser.add_argument('--fold', help='fold\{1,2,..,5\}')
    parser.add_argument('--split', help='\{train, val, test\}')

    args = parser.parse_args()

    print(f"Generating from openness {args.openness}, {args.fold}, {args.split} split")
    start_time = time.time()
    split_dir = os.path.join(args.jams_dir, args.openness, args.fold, args.split)
    print(split_dir)
    create_tag(split_dir, generate_audio=True)
    split_time = time.time() - start_time
    print(f"Generated the split in {split_time} s")
    print("-------------------------------------------------")
