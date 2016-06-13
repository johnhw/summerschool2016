import numpy as np
def window_data(stream,  size, step=None, subsample=None, window=None):
    """Takes a single time series of data `stream` and splits into
    windows of `size`, stepping in increments of `step` each time. 
    The given window function is applied (see below for choices).
    
    Example:
        window_data(audio, 512, step=256, subsample=0.25, window=scipy.signal.hamming)
    
    Parameters:
        stream:         A sequence of values (e.g. audio data)
        size:           Size of the window to be processed.
        step:           Skip from the current sample until next window. 
                        If omitted, set to be equal to `size`; must be > 0.
                        Note that if skip<size, then the windows **will be overlapping**.
        subsample:      Proportion of windows to take. 1.0 = all data (default), 0.5 = random half of the data
                        The sequence is randomly subsampled to take this fraction of the possible windows.
        window:         Window function to apply. e.g. pass scipy.signal.hamming or scipy.signal.hann.
                        Default is will be boxcar (no windowing)
                                            
    Returns:
        an NxD matrix of features
    """
    
    i = 0
    n = len(stream)
    chunks = []
    if window is None:
        window_mask = np.ones(size)
    else:
        window_mask = window(size)
        
    if step is None:
        step = size
    if subsample is None:
        subsample = 1.0
    assert(step>0)    
    while i<n:
        if (i+size)<n and np.random.uniform(0,1)<subsample:
            chunks.append(window_mask*stream[i:i+size])
        i += step
    return np.array(chunks)
    