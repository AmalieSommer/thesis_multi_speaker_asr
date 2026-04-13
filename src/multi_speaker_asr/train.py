from multi_speaker_asr.models import Model
from multi_speaker_asr.data import MyDataset
from pyannote.audio import Pipeline

def train():
    dataset = MyDataset("data/raw")
    model = Model()
    # add rest of your training code here

if __name__ == "__main__":
    train()
