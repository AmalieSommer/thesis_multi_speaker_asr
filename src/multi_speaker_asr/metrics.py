from datasets import load_dataset
import itertools
import json
import csv
from jiwer import wer, cer
from multi_speaker_asr.data import clean_transcription
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter

def compute_error_rates(hypothesis: str, reference: str):
    clean_hyp = clean_transcription(hypothesis)
    clean_ref = clean_transcription(reference)

    return {
        'cer': cer(reference=clean_ref, hypothesis=clean_hyp),
        'wer': wer(reference=clean_ref, hypothesis=clean_hyp),
        'reference': reference
    }

def add_error_rate():
    df = pd.read_csv('coral_metadata.csv')
    with open('src/hpc_results/float32_cputhreads12/asr_output_coral_segments_float32.jsonl', 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f]

    for record in records:
        index = df.index[df['id']==record['audio_id']].tolist()[0]
        reference = df.at[index, 'text']
        error_rates = compute_error_rates(record['text'], reference)
        record['error_rate'] = error_rates

    with open('src/hpc_results/float32_cputhreads12/asr_output_coral_segments_float32.jsonl', 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

def get_samples(res_path: str):
    with open(res_path, 'r') as f:
        records = [json.loads(line) for line in f]
    return records
   
def plot_exp1(int8_cer, int8_wer, float32_cer, float32_wer):
    gender_groups = ["Female", "Male"]
    #age_groups = ["18–29", "30–39", "40–49", "50–59", "60+"]

    wer_fp32 = [float32_wer.get('female'), float32_wer.get('male')]
    wer_int8 = [int8_wer.get('female'), int8_wer.get('male')]

    cer_fp32 = [float32_cer.get('female'), float32_cer.get('male')]
    cer_int8 = [int8_cer.get('female'), int8_cer.get('male')]

    #wer_fp32 = [float32_wer.get(1), float32_wer.get(2), float32_wer.get(3), float32_wer.get(4), float32_wer.get(5)]
    #wer_int8 = [int8_wer.get(1), int8_wer.get(2), int8_wer.get(3), int8_wer.get(4), int8_wer.get(5)]

    #cer_fp32 = [float32_cer.get(1), float32_cer.get(2), float32_cer.get(3), float32_cer.get(4), float32_cer.get(5)]
    #cer_int8 = [int8_cer.get(1), int8_cer.get(2), int8_cer.get(3), int8_cer.get(4), int8_cer.get(5)]

    x = np.arange(len(gender_groups))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
        ax.set_axisbelow(True)

    # WER
    axes[0].bar(x - width/2, wer_fp32, width, label="Float32")
    axes[0].bar(x + width/2, wer_int8, width, label="Int8")
    axes[0].set_ylim(0, 0.5)
    axes[0].set_title("WER")
    axes[0].set_ylabel("WER (%)")
    axes[0].set_xlabel("Gender")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(gender_groups)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(frameon=False)

    # CER
    axes[1].bar(x - width/2, cer_fp32, width)
    axes[1].bar(x + width/2, cer_int8, width)
    axes[1].set_ylim(0, 0.5)
    axes[1].set_title("CER")
    axes[1].set_ylabel("CER (%)")
    axes[1].set_xlabel("Gender")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(gender_groups)
    axes[1].grid(axis="y", alpha=0.3)

    plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    })
    plt.tight_layout()
    plt.show()

    
if __name__=='__main__':
    age_buckets = [18, 29, 39, 49, 59, 60] # Buckets: [<18], [18-29], [30-39], [40-49], [50, 59], [60+]
    df = pd.read_csv('coral_metadata.csv')
    with open('src/hpc_results/int8_cputhreads12/asr_output_coral_segments.jsonl', 'r') as file:
        int8_res = [json.loads(f_line) for f_line in file]

    int8_errs = []
    for i in int8_res:
        id = i['audio_id']
        error_rate = i['error_rate']
        sample = df[df['id'] == id]
        age = sample['age'].values[0]
        gender = sample['gender'].values[0]
        if age >= age_buckets[-1]:
            bucket = len(age_buckets) - 1
        else:
            bucket = [age_buckets.index(ex) for ex in age_buckets if age <= ex][0]
        int8_errs.append({
            'id': id,
            'age': age,
            'age_bucket': bucket,
            'gender': gender,
            'wer': error_rate['wer'],
            'cer': error_rate['cer']
        })

    int8_errs_gender = sorted(int8_errs, key=itemgetter("gender"))
    int8_wer_gender = {}
    int8_cer_gender = {}
    for key, group in itertools.groupby(int8_errs_gender, lambda x: x['gender']):
        group = list(group)
        group_wer = [g['wer'] for g in group]
        if len(group_wer) < 1:
            continue
        int8_wer_gender[key] = sum(group_wer) / len(group_wer)

        group_cer = [g['cer'] for g in group]
        if len(group_cer) < 1:
            continue
        int8_cer_gender[key] = sum(group_cer) / len(group_cer)



    int8_errs_ages = sorted(int8_errs, key=itemgetter("age_bucket"))
    int8_wer_ages = {}
    int8_cer_ages = {}
    for key, group in itertools.groupby(int8_errs_ages, lambda x: x['age_bucket']):
        group = list(group)
        group_wer = [g['wer'] for g in group]
        if len(group_wer) < 1:
            continue
        int8_wer_ages[key] = sum(group_wer) / len(group_wer)
        group_cer = [g['cer'] for g in group]
        if len(group_cer) < 1:
            continue
        int8_cer_ages[key] = sum(group_cer) / len(group_cer)

    with open('src/hpc_results/float32_cputhreads12/asr_output_coral_segments_float32.jsonl', 'r') as file:
        float32_res = [json.loads(f_line) for f_line in file]

    float32_errs = []
    for k in float32_res:
        id = k['audio_id']
        error_rate = k['error_rate']
        sample = df[df['id'] == id]
        age = sample['age'].values[0]
        gender = sample['gender'].values[0]
        if age >= age_buckets[-1]:
            bucket = len(age_buckets) - 1
        else:
            bucket = [age_buckets.index(ex) for ex in age_buckets if age <= ex][0]
        float32_errs.append({
            'id': id,
            'age': age,
            'age_bucket': bucket,
            'gender': gender,
            'wer': error_rate['wer'],
            'cer': error_rate['cer']
        })

    float32_errs_gender = sorted(float32_errs, key=itemgetter("gender"))
    float32_wer_gender = {}
    float32_cer_gender = {}
    for key, group in itertools.groupby(float32_errs_gender, lambda x: x['gender']):
        group = list(group)
        group_wer = [g['wer'] for g in group]
        if len(group_wer) < 1:
            continue
        float32_wer_gender[key] = sum(group_wer) / len(group_wer)

        group_cer = [g['cer'] for g in group]
        if len(group_cer) < 1:
            continue
        float32_cer_gender[key] = sum(group_cer) / len(group_cer)


    float32_errs_ages = sorted(float32_errs, key=itemgetter("age_bucket"))
    float32_wer_ages = {}
    float32_cer_ages = {}
    for key, group in itertools.groupby(float32_errs_ages, lambda x: x['age_bucket']):
        group = list(group)
        group_wer = [g['wer'] for g in group]
        if len(group_wer) < 1:
            continue
        float32_wer_ages[key] = sum(group_wer) / len(group_wer)

        group_cer = [g['cer'] for g in group]
        if len(group_cer) < 1:
            continue
        float32_cer_ages[key] = sum(group_cer) / len(group_cer)


    plot_exp1(int8_cer_gender, int8_wer_gender, float32_cer_gender, float32_wer_gender)
    