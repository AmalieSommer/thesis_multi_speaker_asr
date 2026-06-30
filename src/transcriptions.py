import pandas as pd
import os
from docx import Document
import re
import json
import numpy as np
from datetime import datetime, time, timedelta
import librosa


PATH = os.path.join(os.getcwd(), 'data', 'audio')

def read_file(path: str):
    try:
        doc = Document(path)
    except Exception as e:
        print(f'An error occurred: {e}')
    finally:
        return doc

def extractNumberFiles(doc: Document):
    filesMap = {}

    for i, paragraph in enumerate(doc.paragraphs):
        filenames = paragraph.text.split(':')[1]

        if '+' in filenames:
            filename = filenames.split('+')[0].strip()
            base = filename[:-1] if filename[-1].isdigit() else filename
            num_files = len(filenames.split('+'))

            for i in range(num_files):
                id = i + 1
                filesMap[id] = base + str(id)
        else:
            lst = [filenames]
            id = len(lst)
            filesMap[id] = filenames

        return filesMap

def get_seconds(timestamp: str):
    m, s = timestamp.split(':')
    return float(m) * 60.0 + float(s)  


def generateSegments(doc: Document, filesMap: dict):
    jsonFile = []
    json_line = {}
    transcript_segments = []
    timestamp_pattern = r'\d\d:\d\d' # TODO: Change later for real timestamp pattern...

    flag_ignore_paragraphs = False
    for key, value in filesMap.items():
        json_line['id'] = value # To save the file id of the transcription being passed.

        for p in doc.paragraphs:
            
            match = re.match(r'Fil\s+(\d+)\s+begynder', p.text)
            transcript = {}
                
            if flag_ignore_paragraphs:
                # Should ignore paragraphs until reset...
                if match:
                    if str(key) in match.group():
                        flag_ignore_paragraphs = False
                        continue
            else:
                # Should include paragraphs as normal:

                if match:
                    # Check if the matched line contains the number of the file currently viewed,
                    # otherwise set the ignoring flag to True
                    if str(key) not in match.group():
                        flag_ignore_paragraphs = True
                        continue # ensure the flag is set correctly to not ignore and move onto next paragraph
        
                match_time = re.search('^' + timestamp_pattern, p.text)
                text = p.text
                if match_time:
                    time = match_time.group()
                    transcript['start'] = get_seconds(time)
                    text = re.sub(time, '', text)

                    match_speaker = re.search('(?<=.)[BIV2]:', p.text)
                    if match_speaker:
                        speaker_group = match_speaker.group()
                        transcript['speaker'] = speaker_group if type(speaker_group) == str else speaker_group[0]   # to not include the semicolon...
                        
                        text = re.sub(speaker_group, '', text)
                        transcript['text'] = text

                if len(transcript) != 0:
                    transcript_segments.append(transcript)

        json_line['segments'] = transcript_segments
        
        jsonFile.append(json_line)
        json_line = {}
    return jsonFile

def save(result, filename):
    try:
        with open(os.path.join(os.getcwd(), 'data', f'{filename}.jsonl'), "w") as file:
            for item in result:
                json_line = json.dumps(item)
                file.write(json_line + '\n')
        print(f"Successfully updated the file")
    except Exception as e:
        print(f"Error updating the file: {e}")


if __name__=='__main__':
    print('...')
    metadata = pd.read_excel('data\\Kodningsoverblik - lydfiler.xlsx')
    metadata = metadata.drop(metadata.columns[[1, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]], axis=1)
    metadata.columns = metadata.iloc[0]
    metadata = metadata.drop(metadata.index[0]).reset_index(drop=True)
    metadata = metadata.dropna(subset=['Video title'])
    metadata = metadata.sort_index().reset_index(drop=True)

    for i, row in metadata.iterrows():
        erp_start = metadata.at[i, 'ERP start']
        if type(erp_start) == str:
            erp_start_list = erp_start.split(':')
            if len(erp_start_list) >= 2:
                temp = erp_start_list[-2:]
                metadata.at[i, 'ERP start'] = timedelta(minutes=float(temp[0][-2:]), seconds=float(temp[1][-2:]))
        erp_stop = metadata.at[i, 'ERP stop']
        if type(erp_stop) == str:
            erp_stop_list = erp_stop.split(':')
            if len(erp_stop_list) >= 2:
                temp = erp_stop_list[-2:]
                metadata.at[i, 'ERP stop'] = timedelta(minutes=float(temp[0][-2:]), seconds=float(temp[1][-2:]))


    for i, row in metadata.iterrows():
        id = row['ID']
        if id is np.nan:
            prev_id = metadata.at[i-1, 'ID']
            metadata.at[i, 'ID'] = prev_id

    metadata_2 = pd.DataFrame(columns=['ID', 'BORIS', 'Video title', 'ERP start', 'ERP stop'])
    new_video_titles = []
    for i, row in metadata.iterrows():
        videos = row['Video title'].split('\n')
        video_title_base = videos[0].split('.')[0]
        video_title_base = video_title_base.split(':')[1].strip()
        last_char = video_title_base[len(video_title_base)-1]
        if last_char.isnumeric():
            video_title_base = video_title_base[:len(video_title_base)-1]
            video_range = [item.split('.')[0][-1] for item in videos]
            numbers_range = [int(item) for item in video_range]
            video_range = range(numbers_range[0], numbers_range[1]+1)
            
            new_video_titles = [video_title_base + str(num) + '.flac' for num in video_range]
            for j in range(len(new_video_titles)):
                new_row = pd.DataFrame({
                    'ID': metadata.at[i, 'ID'],
                    'BORIS': metadata.at[i, 'BORIS'],
                    'Session': metadata.at[i, 'Session'],
                    'Video title': new_video_titles[j],
                    'ERP start': metadata.at[i, 'ERP start'],
                    'ERP stop': metadata.at[i, 'ERP stop']
                }, index=[0])
                metadata_2 = pd.concat([metadata_2, new_row], ignore_index=True)
                metadata_2 = metadata_2.sort_index().reset_index(drop=True)

            print(metadata.head(8))
            print(metadata_2.head(8))
        else:
            new_row = pd.DataFrame({
                    'ID': metadata.at[i, 'ID'],
                    'BORIS': metadata.at[i, 'BORIS'],
                    'Session': metadata.at[i, 'Session'],
                    'Video title': video_title_base + '.flac',
                    'ERP start': metadata.at[i, 'ERP start'],
                    'ERP stop': metadata.at[i, 'ERP stop']
                }, index=[0])
            metadata_2 = pd.concat([metadata_2, new_row], ignore_index=True)
            metadata_2 = metadata_2.sort_index().reset_index(drop=True)

        print(videos)

    for index, row in metadata_2.iterrows():

        if index == len(metadata_2) - 2:
            # The timestamps are not provided, so assume they match what is entered in the .docx transcripts:
            metadata_2.at[index, 'ERP start'] = timedelta(seconds=15)
            wav, _ = librosa.load(path=os.path.join(PATH, row['Video title']), sr=16000)
            metadata_2.at[index, 'ERP stop'] = timedelta(seconds=float(wav.shape[0] / 16000))
            continue
        elif index == len(metadata_2) - 1:
            metadata_2.at[index, 'ERP start'] = timedelta()
            metadata_2.at[index, 'ERP stop'] = timedelta(seconds=40)
            continue

        start = row['ERP start']
        end = row['ERP stop']

        if index - 1 >= 0:
            prev_file_id = metadata_2.at[index - 1, 'BORIS']
        curr_file_id = metadata_2.at[index, 'BORIS']
        next_file_id = metadata_2.at[index + 1, 'BORIS']

        if curr_file_id == next_file_id:
            if index == 0:
                # First row...
                wav, _ = librosa.load(path=os.path.join(PATH, row['Video title']), sr=16000)
                metadata_2.at[index, 'ERP stop'] = timedelta(seconds=float(wav.shape[0] / 16000))
            elif prev_file_id == curr_file_id:
                wav, _ = librosa.load(path=os.path.join(PATH, row['Video title']), sr=16000)
                metadata_2.at[index, 'ERP start'] = timedelta()
                metadata_2.at[index, 'ERP stop'] = timedelta(seconds=float(wav.shape[0] / 16000))
            elif prev_file_id != curr_file_id:
                wav, _ = librosa.load(path=os.path.join(PATH, row['Video title']), sr=16000)
                metadata_2.at[index, 'ERP stop'] = timedelta(seconds=float(wav.shape[0] / 16000))
        elif curr_file_id != next_file_id:
            if prev_file_id == curr_file_id:
                metadata_2.at[index, 'ERP start'] = timedelta()
        
    print(metadata_2)
    metadata_2['audio_id'] = metadata_2['Video title'].str.split('.').str[0]
    metadata_2 = metadata_2.rename(columns={'ID': 'patient_id', 'Video title': 'path', 'ERP start': 'start', 'ERP stop': 'stop'})
    metadata_2.to_csv('data\\wrist_angel_metadata.csv', index=False)

    """
    filename = 'transcripts'    # To hold the complete transcripts from all recordings

    files_list = os.listdir('data')
    total_results = []
    # Iterate the list of docx files in the data folder:
    for document in files_list:
        if '.docx' in document:
            path = os.path.join(os.getcwd(), 'data', document)
            doc = read_file(path=path)
            map = extractNumberFiles(doc)
            results = generateSegments(doc, map)
            total_results += results
    
    save(result=total_results, filename=filename)
    """