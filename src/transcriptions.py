import pandas as pd
import os
from docx import Document
import re


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
            id = len(list(filenames))
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
    for p in doc.paragraphs:
        match = re.match(r'Fil\s+(\d+)\s+begynder', p.text)
        transcript = {}
                
        if match:
            # Check if I need to add a pre-existing file to the transcripts list:
            if len(json_line) != 0:
                json_line['segments'] = transcript_segments
                jsonFile.append(json_line)
                json_line = {}

            filenumber = [num for num in p.text if num.isdigit()][0]
            json_line['id'] = filesMap[int(filenumber)]
            transcript_segments = []
        
        match_time = re.search('^' + timestamp_pattern, p.text)
        if match_time:
            transcript['text'] = p.text

            time = match_time.group()
            transcript['start'] = get_seconds(time)
        
        match_speaker = re.search('(?<=.)[BIV]:', p.text)
        if match_speaker:
            transcript['speaker'] = match_speaker.group()[0] # to not include the semicolon...

        if len(transcript) != 0:
            transcript_segments.append(transcript)

    json_line['segments'] = transcript_segments
    jsonFile.append(json_line)
    json_line = {}
    return jsonFile


if __name__=='__main__':
    print('...')
    path = os.path.join(os.getcwd(), 'data\\id4_baseline_exposure1 + 2.docx')
    doc = read_file(path=path)
    map = extractNumberFiles(doc)
    generateSegments(doc, map)

    # Generate function to save list of json as .jsonl file format
