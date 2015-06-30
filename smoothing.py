import numpy
import numpy as np
from math import factorial

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
   3     The Savitzky-Golay filter removes high frequency noise from data.
   4     It has the advantage of preserving the original shape and
   5     features of the signal better than other types of filtering
   6     approaches, such as moving averages techniques.
   7     Parameters
   8     ----------
   9     y : array_like, shape (N,)
  10         the values of the time history of the signal.
  11     window_size : int
  12         the length of the window. Must be an odd integer number.
  13     order : int
  14         the order of the polynomial used in the filtering.
  15         Must be less then `window_size` - 1.
  16     deriv: int
  17         the order of the derivative to compute (default = 0 means only smoothing)
  18     Returns
  19     -------
  20     ys : ndarray, shape (N)
  21         the smoothed signal (or it's n-th derivative).
  22     Notes
  23     -----
  24     The Savitzky-Golay is a type of low-pass filter, particularly
  25     suited for smoothing noisy data. The main idea behind this
  26     approach is to make for each point a least-square fit with a
  27     polynomial of high order over a odd-sized window centered at
  28     the point.
  29     Examples
  30     --------
  31     t = np.linspace(-4, 4, 500)
  32     y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
  33     ysg = savitzky_golay(y, window_size=31, order=4)
  34     import matplotlib.pyplot as plt
  35     plt.plot(t, y, label='Noisy signal')
  36     plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
  37     plt.plot(t, ysg, 'r', label='Filtered signal')
  38     plt.legend()
  39     plt.show()
  40     References
  41     ----------
  42     .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
  43        Data by Simplified Least Squares Procedures. Analytical
  44        Chemistry, 1964, 36 (8), pp 1627-1639.
  45     .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
  46        W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
  47        Cambridge University Press ISBN-13: 9780521880688
  48     """

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        print("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
   5
   6     This method is based on the convolution of a scaled window with the signal.
   7     The signal is prepared by introducing reflected copies of the signal
   8     (with the window size) in both ends so that transient parts are minimized
   9     in the begining and end part of the output signal.
  10
  11     input:
  12         x: the input signal
  13         window_len: the dimension of the smoothing window; should be an odd integer
  14         window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
  15             flat window will produce a moving average smoothing.
  16
  17     output:
  18         the smoothed signal
  19
  20     example:
  21
  22     t=linspace(-2,2,0.1)
  23     x=sin(t)+randn(len(t))*0.1
  24     y=smooth(x)
  25
  26     see also:
  27
  28     numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
  29     scipy.signal.lfilter
  30
  31     TODO: the window parameter could be the window itself if an array instead of a string
  32     NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
  33     """

    if x.ndim != 1:
        raise ValueError("Smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
       return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=numpy.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y