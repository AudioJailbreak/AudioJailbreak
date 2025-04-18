

import argparse
import torchaudio
from torchaudio.functional import resample
import os
import pandas as pd
import jsonlines

from LALM.LLaMA_Omni import LLaMA_Omni


def parse_args(input_args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('-model', '--model', default='LLaMA_Omni')
    parser.add_argument('-type', '--type', choices=['no_attack', 'base', 'benign', 'sound_effect', 'music', 'speed'], default='base')
    parser.add_argument('-max_iter', '--max_iter', default=500, type=int)
    parser.add_argument('-lr', '--lr', default=0.001, type=float)
    parser.add_argument('-start', '--start', default=0, type=int)
    parser.add_argument('-end', '--end', default=50, type=int)
    parser.add_argument('-n_run', '--n_run', default=10, type=int)
    parser.add_argument('-model_t', '--model_t', default=None)

    args = parser.parse_args() if input_args is None else parser.parse_args(args=input_args)

    return args


def main(args):

    if args.type == 'no_attack':
        assert args.model_t is None or args.model_t == args.model, 'No attack, should not set args.model_t'
    
    ## load model
    if args.model_t is None:
        args.model_t = args.model
    
    ## load model and focus index
    if args.model_t == 'LLaMA_Omni':
        model = LLaMA_Omni()
        focus_indexs = list(range(50))
    else:
        raise NotImplementedError('Unsupported Model')
    
    ## attack
    if args.type in ['no_attack', 'base']:
        benign_root = 'AdvBench-harmful_behaviors_custom'
        focus_indexs_2 = list(range(len(os.listdir(benign_root))))
        n_focus_indexs_2 = len(focus_indexs_2)
        benign_root_names = sorted(os.listdir(benign_root), key=lambda x: int(x.split('.wav')[0]))
    else:
        raise NotImplementedError('Unsupported Type')
    
    if args.type != 'no_attack':
        wav_root = f"wavs/s1-{args.model}-{args.type}-lr={str(args.lr).replace('.', '_')}-iter={args.max_iter}"
    else:
        wav_root = benign_root
    print(wav_root)
    assert os.path.exists(wav_root)

    os.makedirs('responses', exist_ok=True)
    if args.type == 'no_attack':
        json_file = f"responses/s1-{args.model}-{args.type}.jsonl"
    else:
        json_file = f"responses/s1-{args.model_t}-{args.model}-{args.type}-lr={str(args.lr).replace('.', '_')}-iter={args.max_iter}.jsonl"
    wr = jsonlines.open(json_file, 'a')

    if not model.random:
        args.n_run = 1
    
    ### question & target
    q_t_path = 'harmful_behaviors_custom.csv'
    df = pd.read_csv(q_t_path)
    goals = list(df['goal'])
    targets = list(df['target'])

    for my_idx, (goal, target) in enumerate(zip(goals, targets)):

        if my_idx not in range(args.start, args.end):
            continue

        if not my_idx in focus_indexs:
            continue

        name_idx  = focus_indexs_2[my_idx % n_focus_indexs_2]
        name = benign_root_names[name_idx]

        wav_flag = f'{my_idx}-{name_idx}.wav' if args.type != 'no_attack' else f'{my_idx}.wav'
        wav_file = os.path.join(wav_root, wav_flag)
        if not os.path.exists(wav_file):
            print('Not Exists, skip', wav_file)
            continue

        audio_path = wav_file
        audio, fs = torchaudio.load(audio_path) # float32
        audio = audio.cuda()
        if fs != 16_000:
            audio = resample(audio, fs, 16_000)

        print('#' * 10, my_idx, name_idx, '#' * 10)
        print('Question: ', goal)
        print('Target: ', target)
        print('#' * 20)

        for my_run in range(args.n_run):

            # try:
            response = model.get_rsp(audio)
            print(my_run, response)
            line = {'flag': wav_flag, 'run': my_run, 'tst': goal, 'target': target, 'text_a': response}
            wr.write(line)
            # except (torch.cuda.OutOfMemoryError, RuntimeError):
            #     print('OOM')
            #     line = {'flag': wav_flag, 'run': my_run, 'tst': goal, 'target': target, 'text_a': 'Error'}
            #     wr.write(line)
            #     continue
    
    wr.close()


if __name__ == '__main__':

    args = parse_args()
    main(args)