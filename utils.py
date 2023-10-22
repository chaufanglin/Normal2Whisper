import numpy as np
from numpy.random import randn
import soundfile as sf
from scipy.signal import lfilter
from scipy.signal.windows import hann
from librosa import lpc
import pyworld as pw


def wav2world(x, fs, fft_size=None):
    """Convenience function to do all WORLD analysis steps in a single call.
    In this case only `frame_period` can be configured and other parameters
    are fixed to their defaults. Likewise, F0 estimation is fixed to
    DIO plus StoneMask refinement.
    Parameters
    ----------
    x : ndarray
        Input waveform signal.
    fs : int
        Sample rate of input signal in Hz.
    fft_size : int
        Length of Fast Fourier Transform (in number of samples)
        The resulting dimension of `ap` adn `sp` will be `fft_size` // 2 + 1
    Returns
    -------
    f0 : ndarray
        F0 contour.
    sp : ndarray
        Spectral envelope.
    ap : ndarray
        Aperiodicity.
    t  : ndarray
        Temporal position of each frame.
    """
    f0, t = pw.harvest(x, fs)
    sp = pw.cheaptrick(x, f0, t, fs, fft_size=fft_size)
    ap = pw.d4c(x, f0, t, fs, fft_size=fft_size)
    return f0, sp, ap, t


def moving_average(data, length):
    output = np.empty(data.shape)
    maf = np.bartlett(length)/length  # Bartlett window is a triangular window
    for i in range(data.shape[0]):
        output[i,:] = np.convolve(data[i,:], maf,'same')
    return output


def gfm_iaif(s_gvl, nv=48, ng=3, d=0.99, win=None):
    """
    Glottal Flow Model-based Iterative Adaptive Inverse Filtering.

    Note:
    Function originally coded by Olivier Perrotin (https://github.com/operrotin/GFM-IAIF). 
    This code is translated to Python and adapted by Chaufang Lin (chaufang.zflin@outlook.com)

    Parameters:
    ----------
        s_gvl: Speech signal frame
        nv: Order of LP analysis for vocal tract (def. 48)
        ng: Order of LP analysis for glottal source (def. 3)
        d: Leaky integration coefficient (def. 0.99)
        win: Window used before LPC (def. Hanning)

    Returns:
    -------
        av: LP coefficients of vocal tract contribution
        ag: LP coefficients of glottis contribution
        al: LP coefficients of lip radiation contribution
    """

    # ----- Set default parameters -------------------------------------------
    if win is None:
        # Window for LPC estimation
        win = np.hanning(len(s_gvl))

    # ----- Addition of pre-frame --------------------------------------------
    # For the successive removals of the estimated LPC envelopes, a
    # mean-normalized pre-frame ramp is added at the beginning of the frame
    # in order to diminish ripple. The ramp is removed after each filtering.
    Lpf = nv + 1  # Pre-frame length
    x_gvl = np.concatenate([np.linspace(-s_gvl[0], s_gvl[0], Lpf), s_gvl])  # Prepend
    idx_pf = np.arange(Lpf, len(x_gvl))  # Indexes that exclude the pre-frame

    # ----- Cancel lip radiation contribution --------------------------------
    # Define lip radiation filter
    al = [1, -d]

    # Integration of signal using filter 1/[1 -d z^(-1)]
    # - Input signal (for LPC estimation)
    s_gv = lfilter([1], al, s_gvl)
    # - Pre-framed input signal (for LPC envelope removal)
    x_gv = lfilter([1], al, x_gvl)

    # ----- Gross glottis estimation -----------------------------------------
    # Iterative estimation of glottis with ng first order filters
    ag1 = lpc(s_gv*win, order=1)         # First 1st order LPC estimation

    for i in range(ng-2):
        # Cancel current estimate of glottis contribution from speech signal
        x_v1x = lfilter(ag1,1,x_gv)        # Inverse filtering
        s_v1x = x_v1x[idx_pf]        # Remove pre-ramp

        # Next 1st order LPC estimation
        ag1x = lpc(s_v1x*win, order=1)        # 1st order LPC

        # Update gross estimate of glottis contribution
        ag1 = np.convolve(ag1,ag1x)        # Combine 1st order estimation with previous


    # ----- Gross vocal tract estimation -------------------------------------
    # Cancel gross estimate of glottis contribution from speech signal
    x_v1 = lfilter(ag1,1,x_gv)       # Inverse filtering
    s_v1 = x_v1[idx_pf]         # Remove pre-ramp

    # Gross estimate of the vocal tract filter
    av1 = lpc(s_v1*win, order=nv)        # nv order LPC estimation

    # ----- Fine glottis estimation ------------------------------------------
    # Cancel gross estimate of vocal tract contribution from speech signal
    x_g1 = lfilter(av1,1,x_gv)       # Inverse filtering
    s_g1 = x_g1[idx_pf]         # Remove pre-ramp

    # Fine estimate of the glottis filter
    ag = lpc(s_g1*win, order=ng)        # ng order LPC estimation

    # ----- Fine vocal tract estimation --------------------------------------
    # Cancel fine estimate of glottis contribution from speech signal
    x_v = lfilter(ag,1,x_gv)       # Inverse filtering
    s_v = x_v[idx_pf]         # Remove pre-ramp

    # Fine estimate of the vocal tract filter
    av = lpc(s_v*win, order=nv)        # nv order LPC estimation


    return av, ag, al


def gfm_iaif_glottal_remove(s_gvl, nv=48, ng=3, d=0.99, win=None):
    """
    Glootal removal function based on GFM-IAIF.

    Note:
    Function originally coded by Olivier Perrotin (https://github.com/operrotin/GFM-IAIF). 
    This code is translated to Python and adapted by Chaufang Lin (chaufang.zflin@outlook.com)

    Parameters:
    ----------
        s_gvl: Speech signal frame
        nv: Order of LP analysis for vocal tract (def. 48)
        ng: Order of LP analysis for glottal source (def. 3)
        d: Leaky integration coefficient (def. 0.99)
        win: Window used before LPC (def. Hanning)

    Returns:
    -------
        s_v: Speech signal with glottis contribution cancelled 
    """

    # ----- Set default parameters -------------------------------------------
    if win is None:
        # Window for LPC estimation
        win = np.hanning(len(s_gvl))

    # ----- Addition of pre-frame --------------------------------------------
    # For the successive removals of the estimated LPC envelopes, a
    # mean-normalized pre-frame ramp is added at the beginning of the frame
    # in order to diminish ripple. The ramp is removed after each filtering.
    Lpf = nv + 1  # Pre-frame length
    x_gvl = np.concatenate([np.linspace(-s_gvl[0], s_gvl[0], Lpf), s_gvl])  # Prepend
    idx_pf = np.arange(Lpf, len(x_gvl))  # Indexes that exclude the pre-frame

    # ----- Cancel lip radiation contribution --------------------------------
    # Define lip radiation filter
    al = [1, -d]

    # Integration of signal using filter 1/[1 -d z^(-1)]
    # - Input signal (for LPC estimation)
    s_gv = lfilter([1], al, s_gvl)
    # - Pre-framed input signal (for LPC envelope removal)
    x_gv = lfilter([1], al, x_gvl)

    # ----- Gross glottis estimation -----------------------------------------
    # Iterative estimation of glottis with ng first order filters
    ag1 = lpc(s_gv*win, order=1)         # First 1st order LPC estimation

    for i in range(ng-2):
        # Cancel current estimate of glottis contribution from speech signal
        x_v1x = lfilter(ag1,1,x_gv)        # Inverse filtering
        s_v1x = x_v1x[idx_pf]        # Remove pre-ramp

        # Next 1st order LPC estimation
        ag1x = lpc(s_v1x*win, order=1)        # 1st order LPC

        # Update gross estimate of glottis contribution
        ag1 = np.convolve(ag1,ag1x)        # Combine 1st order estimation with previous


    # ----- Gross vocal tract estimation -------------------------------------
    # Cancel gross estimate of glottis contribution from speech signal
    x_v1 = lfilter(ag1,1,x_gv)       # Inverse filtering
    s_v1 = x_v1[idx_pf]         # Remove pre-ramp

    # Gross estimate of the vocal tract filter
    av1 = lpc(s_v1*win, order=nv)        # nv order LPC estimation

    # ----- Fine glottis estimation ------------------------------------------
    # Cancel gross estimate of vocal tract contribution from speech signal
    x_g1 = lfilter(av1,1,x_gv)       # Inverse filtering
    s_g1 = x_g1[idx_pf]         # Remove pre-ramp

    # Fine estimate of the glottis filter
    ag = lpc(s_g1*win, order=ng)        # ng order LPC estimation

    # ----- Fine vocal tract estimation --------------------------------------
    # Cancel fine estimate of glottis contribution from speech signal
    x_v = lfilter(ag,1,x_gv)       # Inverse filtering
    s_v = x_v[idx_pf]         # Remove pre-ramp

    return s_v


def pesudo_whisper_gen(s_n, fs, Lv=16):
    """
    Pesudo whispered speech generating function, using GFM-IAIF and moving averge filtering.

    Note:
    This code is written by Chaufang Lin (chaufang.zflin@outlook.com)

    Parameters:
    ----------
        s_n: Normal speech wavform 
        fs: Sample rate
        Lv: order of LP analysis for vocal tract (default: 16)

    Returns:
    -------
        y_pw: Pesudo whispered speech wavform
    """

    EPSILON = 1e-8

    # Overlapp-add (OLA) method
    nfft = pw.get_cheaptrick_fft_size(fs)
    win_length = int(30*fs/1000) # 30ms * fs / 1000
    nhop = round(win_length / 2)
    window = np.hamming(win_length)
    nframes = int(np.ceil(s_n.size / nhop))

    s_gfm = np.zeros(s_n.shape)     # allocate output speech without glottal source

    for n in range(nframes):
        startPoint = n * nhop     # starting point of windowing
        if startPoint + win_length > s_n.size:
            s_gfm[startPoint - nhop + win_length: ] = EPSILON
            continue
        else:
            sn_frame = s_n[startPoint : startPoint+win_length] * window

        s_gfm_frame = gfm_iaif_glottal_remove(sn_frame, Lv)

        s_gfm[startPoint: startPoint + win_length] = s_gfm[startPoint: startPoint + win_length] + s_gfm_frame

    # Extract GFM
    f0_gfm, sp_gfm, ap_gfm, _ = wav2world(s_gfm, fs)

    # Moving Averge Filtering
    maf_freq = 400  # 400 Hz
    maf_w_len = round(maf_freq/fs * nfft)    # 400 Hz
    sp_maf = moving_average(sp_gfm, maf_w_len)

    # Zero F0 and unit Ap
    f0_zero = np.zeros(f0_gfm.shape) + EPSILON
    ap_unit = np.ones(ap_gfm.shape) - EPSILON

    y_pw = pw.synthesize(f0_zero, sp_maf, ap_unit, fs, pw.default_frame_period)

    return y_pw


def glottal_remove_gen(s_n, fs, Lv=16):
    """
    Speech without glottal contribution generating function, using GFM-IAIF.

    Note:
    This code is written by Chaufang Lin (chaufang.zflin@outlook.com)

    Parameters:
    ----------
        s_n: Normal speech wavform 
        fs: Sample rate
        Lv: order of LP analysis for vocal tract (default: 16)

    Returns:
    -------
        y_no_glottal: Speech wavform without glottal contribution
    """

    EPSILON = 1e-8

    # Overlapp-add (OLA) method
    nfft = pw.get_cheaptrick_fft_size(fs)
    win_length = int(30*fs/1000) # 30ms * fs / 1000
    nhop = round(win_length / 2)
    window = np.hamming(win_length)
    nframes = int(np.ceil(s_n.size / nhop))

    s_gfm = np.zeros(s_n.shape)     # allocate output speech without glottal source

    for n in range(nframes):
        startPoint = n * nhop     # starting point of windowing
        if startPoint + win_length > s_n.size:
            s_gfm[startPoint - nhop + win_length: ] = EPSILON
            continue
        else:
            sn_frame = s_n[startPoint : startPoint+win_length] * window

        s_gfm_frame = gfm_iaif_glottal_remove(sn_frame, Lv)

        s_gfm[startPoint: startPoint + win_length] = s_gfm[startPoint: startPoint + win_length] + s_gfm_frame

    # Extract GFM
    f0_gfm, sp_gfm, ap_gfm, _ = wav2world(s_gfm, fs)
    # Zero F0 and unit Ap
    f0_zero = np.zeros(f0_gfm.shape) + EPSILON
    ap_unit = np.ones(ap_gfm.shape) - EPSILON

    y_no_glottal = pw.synthesize(f0_zero, sp_gfm, ap_unit, fs, pw.default_frame_period)

    return y_no_glottal


def bandwidth_widen_gen(s_n, fs, maf_freq=400):
    """
    Speech with expanded formant bandwidth generating function, using moving averge filtering.

    Note:
    This code is written by Chaufang Lin (chaufang.zflin@outlook.com)

    Parameters:
    ----------
        s_n: Normal speech wavform 
        fs: Sample rate
        maf_freq: Moving Averge Filtering window length (default: 400 Hz)

    Returns:
    -------
        y_bandwidth: Speech waveform with expanded formant bandwidth
    """

    # Extract normal speech
    f0_n, sp_n, ap_n, _ = wav2world(s_n, fs)

    # Moving Averge Filtering
    nfft = pw.get_cheaptrick_fft_size(fs)
    maf_w_len = round(maf_freq/fs * nfft)    # 400 Hz
    sp_maf = moving_average(sp_n, maf_w_len)

    y_bandwidth = pw.synthesize(f0_n, sp_maf, ap_n, fs, pw.default_frame_period)

    return y_bandwidth


def lpcfit(x, p=12, h=128, w=None, ov=1):
    """
    Fit LPC to short-time segments.

    Note:
    Function originally coded by Dan Ellis (http://labrosa.org/~dpwe/resources/matlab/polewarp/lpcfit.m). 
    This code is translated to Python and adapted by Chaufang Lin (chaufang.zflin@outlook.com)

    Parameters:
    ----------
        x: a stretch of signal
        p: LPC prder (default: 12)
        h: hopping size (default: 128)
        w: window size (default: 2*h)
        ov: overlap-add parameter (default: 1)
    
    Returns:
    -------
        a: successive all-pole coefficients in rows
        g: per-frame gains
        e: residual excitation
    """
    if w is None:
        w = 2*h

    if x.ndim == 1:
        x = x[np.newaxis, :]

    npts = x.shape[1]
    nhops = npts // h

    # Pad x with zeros so that we can extract complete w-length windows
    # from it
    x = np.pad(x, ((0, 0), ((w-h)//2, (w-h)//2 + h%2)), mode='constant')

    a = np.zeros((nhops, p+1))
    g = np.zeros(nhops)
    if ov == 0:
        e = np.zeros(npts)
    else:
        e = np.zeros((nhops-1)*h+w)

    # Pre-emphasis
    pre = np.array([1, -0.9])
    x = lfilter(pre, 1, x, axis=1)

    for hop in range(nhops):
        # Extract segment of signal
        xx = x[:, hop*h : hop*h+w]
        # Apply hanning window
        wxx = xx * np.hanning(w)[np.newaxis, :]
        # Form autocorrelation (calculates *way* too many points)
        rxx = np.correlate(wxx[0], wxx[0], mode='full')
        # extract just the points we need (middle p+1 points)
        rxx = rxx[w+w//2-p-1:w+w//2+1]
        # Setup the normal equations
        R = np.toeplitz(rxx[:p])
        # Solve for a (horribly inefficient to use full inv())
        an = np.linalg.solve(R, rxx[1:p+1])
        # Calculate residual by filtering windowed xx
        aa = np.concatenate(([1], -an))
        if ov == 0:
            rs = lfilter(aa, 1, xx[0, (w-h)//2:(w+h)//2])
        else:
            rs = lfilter(aa, 1, wxx[0])
        G = np.sqrt(np.mean(rs**2))
        # Save filter, gain and residual
        a[hop,:] = aa
        g[hop] = G
        if ov == 0:
            e[hop*h : (hop+1)*h] = rs / G
        else:
            e[hop*h : hop*h+w] += rs / G

    # Throw away first (win-hop)/2 pts if in overlap mode
    # for proper synchronization of resynth
    if ov != 0:
        e = e[(w-h)//2:]

    return a, g, e


def warppoles(a, alpha):
    """
    warp an all-pole polynomial by substitution

    Parameters:
    ----------
        a (numpy.ndarray): all-pole polynomial defined by rows of a.
        alpha (float): first-order warp factor. Negative alpha shifts poles up in frequency.

    Returns:
    -------
        tuple: polynomials have zeros too, hence B and A.
        B (numpy.ndarray): warped all-pole polynomial.
        A (numpy.ndarray): zeros polynomial.

    """
    # Construct z-hat^-1 polynomial
    d = np.array([-alpha, 1])
    c = np.array([1, -alpha])

    nrows, order = a.shape

    A = np.zeros((nrows, order))
    B = np.zeros((nrows, order))

    B[:, 0] = a[:, 0]
    A[:, 0] = np.ones(nrows)

    dd = d
    cc = c

    # This code originally mapped zeros.  I adapted it to map
    # poles just by interchanging b and a, then swapping again at the
    # end.  Sorry that makes the variables confusing to read.
    for n in range(1, order):

        for row in range(nrows):

            # add another factor to num, den
            B[row, :order] = np.convolve(B[row, :order-1], c)

        # accumulate terms from this factor
        B[:, :len(dd)] = B[:, :len(dd)] + np.multiply(a[:, n], dd)

        dd = np.convolve(dd, d)
        cc = np.convolve(cc, c)

    # Construct the uniform A polynomial (same for all rows)
    AA = np.ones(1)
    for n in range(2, order+1):
        AA = np.convolve(AA, c)

    A = np.tile(AA, (nrows, 1))

    # Exchange zeros and poles
    T = np.copy(A)
    A = np.copy(B)
    B = np.copy(T)

    return B, A


def lpcsynth(a, g, e=[], h=128, ov=1):
    """
    Resynthesize from LPC representation.

    Each row of a is an LPC fit to a h-point (non-overlapping) 
    frame of data.  g gives the overall gains for each frame and 
    e is an excitation signal (if e is empty, white noise is used; 
    if e is a scalar, a pulse train is used with that period).
    ov nonzero selects overlap-add of reconstructed 
    windows, else e is assumed to consist of independent hop-sized 
    segments that will line up correctly without cross-fading
    (matching the ov option to lpcfit; default is ov = 1).
    
    Returns d as the resulting LPC resynthesis.
    """
    if not e:
        e = randn(1, nepts)
    if isinstance(e, (int, float)):
        pd = e
        e = np.zeros(npts)
        e[::pd] = np.sqrt(pd)
    else:
        npts = len(e) - ov * (w - h)
        nepts = len(e)
        
    w = 2 * h
    nhops, p = a.shape
    npts = nhops * h
    nepts = npts + ov * (w - h)
    e = np.hstack((e, np.zeros(w)))
    d = np.zeros(npts)

    for hop in range(nhops):
        hbase = (hop - 1) * h
        oldbit = d[hbase : hbase + h]
        aa = a[hop]
        G = g[hop]
        if ov == 0:
            newbit = G * lfilter(aa, 1, e[hbase : hbase + h])
        else:
            newbit = G * lfilter(aa, 1, e[hbase : hbase + w])[:w]
            newbit = oldbit + (hann(w) * newbit)
        if ov == 0:
            d[hbase : hbase + h] = newbit
        else:
            d[hbase : hbase + w] = newbit[:w]

    # De-emphasis (must match pre-emphasis in lpcfit)
    pre = [1, -0.9]
    d = lfilter(pre, 1, d)

    return d
