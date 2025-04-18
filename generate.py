
import argparse
import torch
import torchaudio
from torchaudio.functional import resample
import os
import pandas as pd
from scipy.io.wavfile import write
import numpy as np

from LALM.LLaMA_Omni import LLaMA_Omni


def parse_args(input_args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('-model', '--model', default='LLaMA_Omni')
    parser.add_argument('-type', '--type', choices=['base', 'benign', 'sound_effect', 'music', 'speed'], default='base')
    parser.add_argument('-max_iter', '--max_iter', default=500, type=int)
    parser.add_argument('-lr', '--lr', default=0.001, type=float)
    parser.add_argument('-start', '--start', default=0, type=int)
    parser.add_argument('-end', '--end', default=50, type=int)

    args = parser.parse_args() if input_args is None else parser.parse_args(args=input_args)

    return args


def main(args):

    ## load model and focus index
    if args.model == 'LLaMA_Omni':
        model = LLaMA_Omni()
        focus_indexs = list(range(50))
    else:
        raise NotImplementedError('Unsupported Model')
    
    ## attack
    if args.type == 'base':
        benign_root = 'AdvBench-harmful_behaviors_custom'
        focus_indexs_2 = list(range(len(os.listdir(benign_root))))
        n_focus_indexs_2 = len(focus_indexs_2)
        benign_root_names = sorted(os.listdir(benign_root), key=lambda x: int(x.split('.wav')[0]))
    else:
        raise NotImplementedError('Unsupported Type')
    
    
    wav_root = f"wavs/s1-{args.model}-{args.type}-lr={str(args.lr).replace('.', '_')}-iter={args.max_iter}"
    os.makedirs(wav_root, exist_ok=True)
    
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

        try:
            name_idx  = focus_indexs_2[my_idx % n_focus_indexs_2]
            name = benign_root_names[name_idx]
            audio_path = os.path.join(benign_root, name)
            audio, fs = torchaudio.load(audio_path) # float32
            audio = audio.cuda()
            if fs != 16_000:
                audio = resample(audio, fs, 16_000)
            # audio.requires_grad = True
            z = torch.randn_like(audio, device=audio.device, requires_grad=True) # change of optimized variable which is free of box-constraint

            wav_file = os.path.join(wav_root, f'{my_idx}-{name_idx}.wav')
            if os.path.exists(wav_file):
                print('Exists, skip', wav_file)
                continue

            print('#' * 10, my_idx, name_idx, '#' * 10)
            print('Question: ', goal)
            print('Target: ', target)
            print('#' * 20)

            max_iter = args.max_iter
            lower = -1
            upper = 1
            upper = upper * torch.ones_like(audio, device=audio.device) # no eps
            lower = lower * torch.ones_like(audio, device=audio.device) # no eps

            crit = torch.nn.CrossEntropyLoss()
            opt = torch.optim.Adam([z], lr=args.lr)
            
            expected_response = target
            target_label = model.encode(expected_response)

            for cur_iter in range(max_iter+1):
                
                audio_input = audio + torch.tanh(z)
                audio_input = torch.min(torch.max(audio_input, lower), upper)
                logits, b = model.get_logits_indice(audio_input, expected_response)
                loss = crit(logits[0, b-1:b-1+len(target_label), :], target_label)
                print(cur_iter, loss.item())

                if cur_iter % 100 == 0 or cur_iter == max_iter:
                    response = model.get_rsp(audio_input)
                    print(response)

                if cur_iter < max_iter:
                    loss.backward()
                    opt.step()
                    opt.zero_grad()
            
            audio = audio + torch.tanh(z)
            audio = torch.min(torch.max(audio, lower), upper)
            audio = (audio.detach().cpu().numpy().flatten() * (2 ** (16-1))).astype(np.int16)
            write(wav_file, 16_000, audio)
        
        except (torch.OutOfMemoryError, ):
            print('OOM')
            continue


if __name__ == '__main__':

    args = parse_args()
    main(args)