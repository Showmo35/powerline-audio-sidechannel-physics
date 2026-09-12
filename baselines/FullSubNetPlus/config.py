import os

BASE_DIR = '<REPO_ROOT>'

DATA_FOLDERS = {
    'May29_Alice': os.path.join(BASE_DIR, 'May29_Alice'),
    'July10_Podcasts': os.path.join(BASE_DIR, 'July10_Podcasts'),
}
MP3_DIR = os.path.join(BASE_DIR, 'Alice_In_Wonderland_mp3')

CHAPTER_TIMING = {
    'chapter_01': {'start': 1.38, 'mp3_duration': 631.95},
    'chapter_02': {'start': 2.04, 'mp3_duration': 724.14},
    'chapter_04': {'start': 1.95, 'mp3_duration': 1168.38},
    'chapter_05': {'start': 2.17, 'mp3_duration': 794.54},
    'chapter_06': {'start': 2.09, 'mp3_duration': 765.18},
    'chapter_07': {'start': 2.05, 'mp3_duration': 1027.74},
    'chapter_08': {'start': 1.95, 'mp3_duration': 789.39},
    'chapter_10': {'start': 1.63, 'mp3_duration': 1345.72},
    'chapter_11': {'start': 1.61, 'mp3_duration': 601.23},
}

FS_CAPTURE = 200_000
AUDIO_SR = 16_000

BP_LOW = 50
BP_HIGH = 4000

N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 512

WINDOW_SEC = 4.0
WINDOW_HOP_SEC = 1.0

TRAIN_RATIO = 0.9
SPLIT_GUARD_SEC = 4.0

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DEFAULT_DATA_NPZ = os.path.join(DEFAULT_DATA_DIR, 'train_data_fullsubnetplus.npz')
DEFAULT_CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
DEFAULT_INFER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inference_output')
