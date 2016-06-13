import sklearn.metrics
import scipy.interpolate
import scipy.io.wavfile
import numpy as np
from tqdm import tnrange, tqdm_notebook
from time import sleep

true_labels = [[0, 1, 4],
            [3, 3, 4, 4],
            [1, 3],
            [2, 0, 1, 2, 3],
            [1],
            [4, 2, 4],
            [1, 2],
            [2, 3, 0, 1]]

def test_classification_score(wave_data, labels_true, labels_predicted):    
    sr = 4096
    ts = np.arange(len(wave_data))/float(sr)
    
    # make sure there are at least 2 predictions, so interpolation does not freak out
    if len(labels_predicted)==1:
        labels_predicted = [labels_predicted[0], labels_predicted[0]]
    if len(labels_true)==1:
        labels_true = [labels_true[0], labels_true[0]]
    
    # predict every 5ms
    frames = ts[::40]
    
    true_inter = scipy.interpolate.interp1d(np.linspace(0,np.max(ts),len(labels_true)), labels_true, kind="nearest")
    predicted_inter = scipy.interpolate.interp1d(np.linspace(0,np.max(ts),len(labels_predicted)), labels_predicted, kind="nearest")
        
    true_interpolated =true_inter(frames)[:,None]
    predicted_interpolated = predicted_inter(frames)[:,None]
    
    acc = sklearn.metrics.accuracy_score(true_interpolated, predicted_interpolated)
    # basic accuracy
    
    silence_pred = predicted_interpolated==0
    silence_true = true_interpolated==0
    acc_silence = sklearn.metrics.accuracy_score(silence_true, silence_pred)
    
    
    pre, rec, fb, support = sklearn.metrics.precision_recall_fscore_support(true_interpolated, predicted_interpolated, labels=[0,1,2,3,4])
    
    return acc_silence + 2.0 * acc + np.sum(fb*support)/np.sum(support) + np.tanh(len(labels_predicted)*0.01)
    
def load_wave(basename):
    sr, wave = scipy.io.wavfile.read(basename+".wav") 
    return wave / 32768.0

def challenge_evaluate_performance(fn):
    score = 0    
    for i in tnrange(8, desc="Total"):
        wave = load_wave("data/secret_tests/challenge_valid_%d"%i)    
        labels = true_labels[i]
        pred_labels = fn(wave)
        for j in range(3):
            # best of 3!
            score += test_classification_score(wave, labels, pred_labels)
        
        for j in tqdm_notebook(xrange(40), desc='Test case %d'%i):
            sleep(0.1)
    print "*** Total score: %.2f ***" % score
    return score
        
      