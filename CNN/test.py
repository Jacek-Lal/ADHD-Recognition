from config import *
from eeg_read import *
from plots import *
import numpy as np
ADHD_DATA, CONTROL_DATA = readEEGRaw(EEG_DATA_PATH)
ADHD_FILTERED, CONTROL_FILTERED = filterEEGData(ADHD_DATA, CONTROL_DATA, 2)

patient = 0
channel = 0
treshold = 99.5

percentile = np.percentile(ADHD_DATA[patient][channel], treshold)
data_length = ADHD_DATA[patient][channel].shape[0]
t = [x/FS for x in range(data_length)]
plot(ADHD_DATA, patient, channel)
plt.plot(t, [percentile for x in range(data_length)])
plt.plot(t, [-percentile for x in range(data_length)])
plt.show()