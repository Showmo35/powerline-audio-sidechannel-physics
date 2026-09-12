import sys, soundfile as sf, numpy as np
sys.path.insert(0,'<REPO_ROOT>/soundbar_setup/M10_new_setup_soundbar')
from asr import wer
import torch, whisper
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
A='<REPO_ROOT>/soundbar_setup/M14_new_setup_soundbar_melgen/outputs/cluster_a100/audio'
def tr(m,f):
    x,sr=sf.read(f,dtype='float32'); assert sr==16000
    return m.transcribe(np.ascontiguousarray(x),language='en',fp16=(dev=='cuda'),verbose=False)['text'].strip()
for name in ('small','large-v3'):
    m=whisper.load_model(name,device=dev)
    tg=tr(m,f'{A}/chunk_096_670s_target.wav'); gn=tr(m,f'{A}/chunk_096_670s_generated.wav')
    w=wer(tg,gn)
    print(f'== whisper-{name} ==')
    print('  TARGET (GL of REAL mel) :',repr(tg))
    print('  GENERATED (step-2000)   :',repr(gn))
    print(f'  WER(gen vs target)={w["wer"]*100:.1f}%  (S{w["sub"]} D{w["del"]} I{w["ins"]}/{w["ref_words"]}w)')
