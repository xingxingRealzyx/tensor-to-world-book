"""Infer from an explicit controlled prompt or a file in TinyOmni's input domain.

This deliberately does not claim support for natural speech or arbitrary photos.
Text grammar: '<color> <shape> <left|right>'. Image: 16x16 RGB. Audio:
mono 16-bit PCM WAV, 8000 Hz, exactly 256 samples of the synthetic tone protocol.
"""
import argparse
import json
from pathlib import Path
import wave
import numpy as np
from PIL import Image
import torch
from .dataset import VOCAB,ID_TO_TOKEN,COLORS,SHAPES,MOTIONS
from .train import load_checkpoint,choose_device


def make_inputs(task,device):
    ids=[VOCAB['<bos>'],VOCAB[f'<task_{task}>']]+[VOCAB['<unk>']]*3
    return {'text':torch.tensor([ids],device=device),
            'image':torch.zeros(1,3,16,16,device=device),
            'audio':torch.zeros(1,256,device=device),
            'video':torch.zeros(1,4,3,16,16,device=device),
            'presence':torch.zeros(1,3,device=device,dtype=torch.bool)}


def write_audio(path,samples):
    pcm=(np.clip(samples,-1,1)*32767).round().astype('<i2')
    with wave.open(str(path),'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(8000);f.writeframes(pcm.tobytes())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--text');group.add_argument('--image',type=Path);group.add_argument('--audio',type=Path)
    parser.add_argument('--out',type=Path,default=Path('runs/custom-demo'))
    parser.add_argument('--device',default='auto',choices=['auto','cpu','cuda','mps'])
    args=parser.parse_args();device=choose_device(args.device)
    task='text' if args.text is not None else 'image' if args.image else 'audio'
    batch=make_inputs(task,device)
    if task=='text':
        words=args.text.strip().lower().split()
        if len(words)!=3 or words[0] not in COLORS or words[1] not in SHAPES or words[2] not in MOTIONS:
            parser.error('text must be <red|green|blue> <circle|square|triangle> <left|right>')
        batch['text'][0,2:]=torch.tensor([VOCAB[w] for w in words],device=device)
    elif task=='image':
        with Image.open(args.image) as im:
            if im.size!=(16,16):parser.error('image must be exactly 16x16; arbitrary resizing is not supported')
            arr=np.asarray(im.convert('RGB'),dtype=np.float32)/255.
        batch['image'][0]=torch.tensor(arr.transpose(2,0,1),device=device)
        batch['presence'][0,0]=True
    else:
        with wave.open(str(args.audio),'rb') as wav:
            if (wav.getnchannels(),wav.getsampwidth(),wav.getframerate(),wav.getnframes())!=(1,2,8000,256):
                parser.error('audio must be mono PCM16, 8000 Hz, exactly 256 samples')
            arr=np.frombuffer(wav.readframes(256),dtype='<i2').astype(np.float32)/32767.
        batch['audio'][0]=torch.tensor(arr,device=device);batch['presence'][0,1]=True
    model,checkpoint=load_checkpoint(args.checkpoint,device)
    predicted=model.generate(batch)
    tokens=[ID_TO_TOKEN[i] for i in predicted['tokens'][0].cpu().tolist()]
    result={'task':task,'input_text':args.text,'predicted_tokens':tokens,
            'training_step':checkpoint['step'],'ground_truth_available':False,
            'action_available':task=='text',
            'predicted_actions':predicted['action'][0].cpu().tolist() if task=='text' else None,
            'scope':'controlled vocabulary, colored shapes, artificial tones; not natural-language or speech inference'}
    args.out.mkdir(parents=True,exist_ok=True)
    pixels=(predicted['image'][0].permute(1,2,0).cpu().numpy().clip(0,1)*255).round().astype(np.uint8)
    Image.fromarray(pixels).save(args.out/'generated_image.png')
    write_audio(args.out/'generated_audio.wav',predicted['audio'][0].cpu().numpy())
    (args.out/'prediction.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    (args.out/'predicted.txt').write_text(' '.join(t for t in tokens if t not in ('<eos>','<pad>'))+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
