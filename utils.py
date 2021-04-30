from collections import namedtuple
import numpy
import pandas
from scipy.stats import trim_mean
from scipy.stats.mstats import trimmed_std
from scipy.stats import kurtosis, skew
from statsmodels.tsa.stattools import pacf


time_series = namedtuple('time_series', 'uid signal label')

def generate_stationary_series(center, spreads, size):
    N_sources = len(spreads)
    series = numpy.array([numpy.random.normal(loc = center, scale = spreads[i], size = size) for i in range(N_sources)])
    return numpy.mean(series, axis = 0)

def generate_uid():
    return ''.join(numpy.random.choice('a b c d e f g 1 2 3 4 5 6 7 8 9 0'.split(), 9))

def generate_random_walk(size, step):
    res = numpy.zeros(size)
    for j in range(1,size):
        direction = numpy.random.choice([1,-1])
        res[j] = res[j-1] + step*direction
    return res

def generate_drop(size, position, drop):
    res = numpy.zeros(size)
    res[position:] = -drop
    return res

def generate_beam_dump(series):
    size = len(series)
    start_position = numpy.random.randint(0,size)
    end_position = numpy.random.randint(start_position, size)
    series[start_position:end_position] = 0

def generate_high_noise_region(series):
    size = len(series)
    large_level = numpy.max(series)-numpy.min(series)
    start_position = numpy.random.randint(0,size)
    end_position = numpy.random.randint(start_position, size)
    series[start_position:end_position] += generate_stationary_series(0, [large_level], end_position-start_position)

def generate_steady_series(N_samples=500, max_size = 1000):
    steady_list = []
    for j in range(N_samples):
        size = numpy.random.randint(40, max_size+1)
        new_signal = (0.1*generate_random_walk(size=size, step = 0.5)+
                      generate_stationary_series(0, [0.2, 1, 10], size) + 
                      numpy.random.rand()*100+10)
        new_uid = generate_uid()
        new_sample = time_series(new_uid, new_signal, 'steady')
        steady_list.append(new_sample)
    return steady_list

def generate_anomaly_series(N_samples=500, max_size = 1000):
    anomaly_list = []
    for j in range(N_samples):
        size = numpy.random.randint(40, max_size+1)
        random_walk_part = 6*generate_random_walk(size=size, step = 0.5)
        steady_level = numpy.random.rand()*50 
        drop_hight = numpy.random.rand()*steady_level
        drop_position = numpy.random.randint(10, max_size-10)
        new_signal = (generate_drop(size, drop_position, drop_hight)
                     + generate_stationary_series(0, [0.2, 1, 10], size)
                     + random_walk_part
                     + steady_level + (numpy.max(random_walk_part) - numpy.min(random_walk_part)))
        #generate beam dump
        prob_beam_dump = numpy.random.random()
        if prob_beam_dump < 0.1:
            generate_beam_dump(new_signal)
            
        #generate high noise region
        prob_high_noise = numpy.random.random()
        if prob_high_noise < 0.1:
            generate_high_noise_region(new_signal)
            
        new_signal[new_signal<0] = 0
            
        new_uid = generate_uid()
        new_sample = time_series(new_uid, new_signal, 'anomaly')
        anomaly_list.append(new_sample)
        
        
    return anomaly_list



def trim_series(x, p_cut =0.025):
    """ 
    Discards p_cut of the smallest and the largets values from the series
    Returns:
    -------
        redcued series
    """
    N = len(x)
    N_start = int(p_cut*N)
    N_end = int((1-p_cut)*N)
    sorted_x = sorted(x)
    return sorted_x[N_start:N_end]

def autocorr(x, t=1):
    """calculates autocorrelation coefficient with lag t """
    return numpy.corrcoef(numpy.array([x[:-t], x[t:]]))[0,1]

def trimmed_kurtosis(x):
    """ calculate kurtosis for series without extreme values"""
    trimmed_x = trim_series(x)
    return kurtosis(trimmed_x)

def trimmed_skew(x):
    """ calculate skew for series without extreme values"""
    trimmed_x = trim_series(x)
    return skew(trimmed_x)


def generate_all_features(series):
    
    features = pandas.DataFrame()
    
    # calculate the first derivative
    series_diff = series.apply(lambda x: (x[1:] - x[:-1])[1:-1])

    
    # extract standard deviations
    features['sd'] = series.apply(trimmed_std)
    features['diff_sd'] = series_diff.apply(trimmed_std)
    
    for lag in range(1,5):
        # extract correlation coefficients
        features[f'ac_{lag}'] = series.apply(lambda x: autocorr(x, t=lag))
        features[f'diff_ac_{lag}'] = series_diff.apply(lambda x: autocorr(x, t=lag))
        
        # extract partial correlation coefficients
        features[f'pac_{lag}'] = series.apply(lambda x: pacf(x,4)[lag])
        features[f'diff_pac_{lag}'] = series_diff.apply(lambda x: pacf(x,4)[lag])
        
    # calculate difference between beginning and the end
    features['start_end'] = series.apply(lambda x: abs(x[:5].mean()-x[-5:].mean()))
    features['diff_start_end'] = series_diff.apply(lambda x: abs(x[:5].mean()-x[-5:].mean()))
    
    # calculate kurtosis, excluding extreme values
    features['kurtosis'] = series.apply(trimmed_kurtosis)
    features['diff_kurtosis'] = series_diff.apply(trimmed_kurtosis)
    
    # calculate skew, excluding extreme values
    features['skew'] = series.apply(trimmed_kurtosis)
    features['diff_skew'] = series_diff.apply(trimmed_kurtosis)
    
    features['sd_ratio'] = features['sd']/features['diff_sd']
    
    return features