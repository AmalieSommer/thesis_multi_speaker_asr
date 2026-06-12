# This is a script to create a long-form version of the original CoRal-v3 conversation dataset.
# It will be a concatenation of random segments belonging to the same conversation with varying length of pauses.
# Timestamps will be estimated based on the segment durations and varying pauses.
# It should save a csv file with the conversation id, the combined audio file, the speaker(s) included and a list of Segments.
# The Segments will contain [start, end] information based on estimates from concatenating previous segments, as well as information on speaker_id and overlap detection for each Segment.


import pandas as pd
import os
import json
from datasets import load_dataset, Audio
import librosa
import io
import numpy as np
from dataclasses import dataclass, asdict
import dataclasses
import soundfile as sf
from warnings import warn



@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    speaker: str
    overlap: bool

    def _asdict(self):
        warn(
            "Segment._asdict() method is deprecated, use dataclasses.asdict(Segment) instead",
            DeprecationWarning,
            2,
        )
        return asdict(self)




def load_data(path: str):
    ds = None
    if path is None:

        ds = load_dataset(
            path='CoRal-project/coral-v3',
            name='conversation',
            split='test'
        )
    else:
        ds = load_dataset(path=path)

    ds = ds.cast_column('audio', Audio(decode=False))
    ds = ds.rename_column(
        original_column_name='id_conversation',
        new_column_name='id_segment'
    )
    df = ds.to_pandas()
    return df


def generateMap(df: pd.DataFrame):
    # First extract the base id from the id_conversation:
    ids = df['id_segment'].str.split('_', expand=True) #E.g. conv_xxxxxx_0001 --> ['conv', 'xxxxxx', '0001']

    # Generate two new columns with conversation-based information:
    df['id_conversation'] = ids[0] + '_' + ids[1]
    df['index_segment'] = ids[2]

    grouped_segments = df.groupby('id_conversation')['id_segment'].apply(list)
    grouped_dict = grouped_segments.to_dict()

    return grouped_dict


def save_data(filename: str, list: list):
    try:
        with open(os.path.join('/root/master_thesis/thesis_multi_speaker_asr/data/coral-v3-long-form-conversations', f'{filename}_segments.jsonl'), "w") as file:
            for item in list:
                json_line = json.dumps(item)
                file.write(json_line + '\n')
        print(f"Successfully updated the file")
    except Exception as e:
        print(f"Error updating the file: {e}")



def segments_to_long_conversation(df: pd.DataFrame, map: dict, sr=16000):
    conversations = []
    for key, val in map.items():
        segments = val # List of segments in the given conversation...
        total_segments = len(segments)

        start_conversation = 0.0
        end_conversation = 0.0
        prev = 0.0      # running duration counter
        seg_list = []
        combined_audio_list = []
        for index, segment in enumerate(segments):
            row_index = df.index[df['id_segment'] == segment].tolist()[0]

            # Should get the id_speaker, text spoken in the segment, overlap detection boolean, audio bytes
            audio = df.at[row_index, 'audio']
            audio = io.BytesIO(audio['bytes'])
            wav, sr = librosa.load(audio, sr=sr)

            # Generate the silence array (in-between segments pauses)
            length_of_pause = int(sr * np.random.uniform(0.0, 1.5))
            silence_array = np.zeros(length_of_pause)
            
            # Append the noisy pause to the original segment
            combined_audio = np.concat((wav, silence_array), axis=None)
            seg_duration = librosa.get_duration(y=combined_audio, sr=sr)
            combined_audio_list.append(combined_audio) # List of all the concatenated audio arrays.

            # Generate the new Segment:
            start_segment = 0.0
            end_segment = 0.0

            if index == 0: # start of conversation...
                start_segment = 0.0
                end_segment = seg_duration
            else:
                start_segment = prev
                end_segment = start_segment + seg_duration

            seg = Segment(
                id=segment,
                start=start_segment,
                end=end_segment,
                text=df.at[row_index, 'text'],
                speaker=df.at[row_index, 'id_speaker'],
                overlap=1 if df.at[row_index, 'overlap'] else 0     # maps boolean to int in order to save it as json by making it serializable
            )

            prev = prev + seg_duration # move cursor to the next segment time

            if index == total_segments - 1:
                # set the ending time for the whole conversation
                end_conversation = prev

            seg_list.append(dataclasses.asdict(seg))


        full_audio = np.hstack(combined_audio_list)
        full_audio_path = f'/root/master_thesis/thesis_multi_speaker_asr/data/coral-v3-long-form-conversations/{key}.wav'
        sf.write(full_audio_path, full_audio, samplerate=sr, format='wav')

        conversations.append({
            'id': key,
            'start': start_conversation,
            'end': end_conversation,
            'segments': seg_list,
            'path': full_audio_path
        })

    return conversations


if __name__=='__main__':
    df = load_data(path=None)
    mappedGroup = generateMap(df=df)
    long_form_audio = segments_to_long_conversation(df=df, map=mappedGroup)
    save_data(filename='conversation_metadata', list=long_form_audio)