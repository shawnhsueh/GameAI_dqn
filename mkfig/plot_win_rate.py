import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

with open('win_rate.npy', 'rb') as f:
    win_rate = np.load(f)

win_rate = medfilt(win_rate, 11)

plt.plot(np.linspace(0, len(win_rate)*100, len(win_rate)), win_rate)
plt.xlabel('rounds of games')
plt.ylabel('win_rate')
plt.savefig('win_rate.png', bbox_inches='tight')
