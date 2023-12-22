from eeg_read import *
from eeg_filter import *
from config import *
from plots import *

if __name__ == '__main__':
    ADHD_DATA, CONTROL_DATA = readEEGRaw(EEG_DATA_PATH)
    ADHD_FILTERED, CONTROL_FILTERED = filterEEGData(ADHD_DATA, CONTROL_DATA, 2)
    ADHD_CLIPPED, CONTROL_CLIPPED, ADHD_TRESHOLDS, CONTROL_TRESHOLDS = clipEEGData(ADHD_FILTERED, CONTROL_FILTERED)
    #ADHD_NORMALIZED, CONTROL_NORMALIZED = normalizeEEGData(ADHD_CLIPPED, CONTROL_CLIPPED)
    #ADHD_FRAMED = framedEEGData(ADHD_NORMALIZED, EEG_SIGNAL_FRAME_SIZE)
    percentile = 99.8
    for i in range(20):
        for j in range(19):
            patient = i
            channel = j
            treshold = ADHD_TRESHOLDS[i * CNN_INPUT_SHAPE[0] + j]
            plot_with_treshold(ADHD_FILTERED, patient, channel, treshold)
            plt.show()

            plot_with_treshold(ADHD_CLIPPED, patient, channel, treshold)
            plt.show()