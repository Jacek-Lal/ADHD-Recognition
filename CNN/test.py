from eeg_read import *
from config import *
from plots import *

if __name__ == '__main__':
    ADHD_DATA, CONTROL_DATA = readEEGRaw(EEG_DATA_PATH)
    ADHD_NORMALIZED, CONTROL_NORMALIZED = normalizeEEGData(ADHD_DATA, CONTROL_DATA)


    for i in range(len(CUTOFFS)-1):

        ADHD_FILTERED, CONTROL_FILTERED = filterEEGData(ADHD_NORMALIZED, CONTROL_NORMALIZED, i)

        plot_frequency_band(ADHD_FILTERED[0][0], i)

    plt.show()
