import json
import os
import re
import numpy as np
import scipy.stats as stats
from scipy.stats import ttest_rel, iqr, wilcoxon
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



fp32_result_file = '/root/master_thesis/thesis_multi_speaker_asr/src/hpc_results/exp_1/fp32/pipeline_performance.log'
int8_result_file = '/root/master_thesis/thesis_multi_speaker_asr/src/hpc_results/exp_1/int8/pipeline_performance.log'


def get_walltime(filename: str):
    epoch_pattern = r'Epoch: \d+'
    walltime_pattern = r'Walltime: [+-]?([0-9]*[.])?[0-9]+'

    result = []
    current_epoch = None
    epoch_result = []
    with open(filename, 'r') as file:
        for line in file:
            m_epoch = re.search(epoch_pattern, line)
            m_walltime = re.search(walltime_pattern, line)
            if not m_epoch:
                continue
            elif not m_walltime:
                continue

            epoch = str.split(m_epoch.group(), sep=' ')[-1]
            walltime = str.split(m_walltime.group(), sep=' ')[-1]

            if current_epoch is None:
                current_epoch = epoch
                epoch_result.append(float(walltime))
            elif current_epoch != epoch:
                # end of epoch values so save and reset to next epoch...
                result.append(epoch_result)
                epoch_result = [float(walltime)]
                current_epoch = epoch
            else:
                epoch_result.append(float(walltime))
        result.append(epoch_result)

    return np.array(result)


def paired_ttest(data_fp32: np.ndarray, data_int8: np.ndarray, alpha: float = 0.05):

    print(data_fp32.shape)

    if len(data_fp32) != len(data_int8):
        raise Exception('Number of data samples are not equal!')
    
    mean_fp32, mean_int8 = np.average(data_fp32), np.average(data_int8)
    std_fp32, std_int8 = np.std(data_fp32, ddof=1), np.std(data_int8, ddof=1)
    var_fp32, var_int8 = np.var(data_fp32), np.var(data_int8)

    print(f'Mean of FP32: {mean_fp32} and INT8: {mean_int8}')
    print(f'Standard Deviation... FP32: {std_fp32} and for INT8: {std_int8}')

    print(f'Percentage speed-up: {((mean_fp32 - mean_int8)/mean_fp32) * 100}')
    print(f'Variance of FP32: {var_fp32}, and of INT8: {var_int8}')

    n = len(data_fp32)
    degrees_freedom = n-1
    
    diff = data_fp32 - data_int8
    mean_diff = np.average(diff)

    epoch_mean_fp32 = np.mean(data_fp32, axis=1)
    epoch_mean_int8 = np.mean(data_int8, axis=1)

    res = ttest_rel(epoch_mean_fp32,
                    epoch_mean_int8,
                    alternative="greater")
    print(res.statistic)
    print(res.pvalue)


def wilcoxon_ttest(data_fp32: np.ndarray, data_int8: np.ndarray):

    epoch_mean_fp32 = np.mean(data_fp32, axis=1)
    epoch_mean_int8 = np.mean(data_int8, axis=1)

    # Paired differences are implicitly computed as fp32 - int8
    result = wilcoxon(
        epoch_mean_fp32,
        epoch_mean_int8,
        alternative="greater",
        zero_method="wilcox"
    )

    print(f"Wilcoxon statistic: {result.statistic}")
    print(f"p-value: {result.pvalue:.3e}")


if __name__=='__main__':
    data_fp32 = get_walltime(fp32_result_file)
    data_int8 = get_walltime(int8_result_file)

    data_fp32 = np.transpose(data_fp32)
    data_int8 = np.transpose(data_int8)

    paired_ttest(data_fp32=data_fp32, data_int8=data_int8)
    wilcoxon_ttest(data_fp32=data_fp32, data_int8=data_int8)
    
    epoch_avg_fp32 = np.average(data_fp32, axis=1)
    epoch_avg_int8 = np.average(data_int8, axis=1)

    diff = epoch_avg_fp32 - epoch_avg_int8
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)

    cohen = mean_diff / std_diff
    print(f'Mean of difference: {mean_diff}, Standard Deviation: {std_diff} and Cohens d: {cohen}')

    iqr_diff = iqr(diff)
    bin_width = 2 * iqr_diff * ((len(diff))**(-1/3))

    sns.histplot(diff, kde=True, bins=20, binwidth=bin_width, color='purple')
    # Labels and title
    plt.xlabel('X')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()


    