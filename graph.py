import matplotlib.pyplot as plt
from matplotlib import pyplot
#import numpy as np
from numpy.random import random
#### Seed
import random
import numpy as np
seed = 42
np.random.seed(seed)


def bland_altman_plot(data1, data2, name = " ", *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    fig, ax = plt.subplots(figsize=(5,3))
    plt.title('Bland-Altman Plot')
#    plt.legend()
    plt.scatter(mean, diff, *args, **kwargs)

    plt.axhline(md + 1.96*sd, color='g', label="md + 1.96*sd", linestyle='--')
    plt.axhline(md,           color='r', label="md",           linestyle='--')
    plt.axhline(md - 1.96*sd, color='b', label="md - 1.96*sd", linestyle='--')

    labels = ["md + 1.96*sd", "md", "md - 1.96*sd"]
    handles, _ = ax.get_legend_handles_labels()

    # Slice list to remove first handle
    plt.legend(handles = handles[:], labels = labels)
    plt.xlabel("Average(mmHg)",fontsize=10)
    plt.ylabel("Difference(mmHg)",fontsize=10)
    plt.savefig("imgs/"+ str(name)+".pdf", bbox_inches='tight', dpi=320)
    plt.show()
  
def act_pred_plot(y, predicted, R_2=None, mae=None, name = ""):
    fig, ax = plt.subplots(figsize=(5,3))
    ax.text(y.min(), y.max()-3, '$MAE =$ %0.3f' %(np.mean(mae)))
    ax.text(y.min(), y.max()-9.5, '$R^2 =$ %.3f' %(np.mean(R_2)))
    ax.scatter(y, predicted)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r-', lw=1)
    ax.set_xlabel('Reference(mmHg)')
    ax.set_ylabel('Estimated(mmHg)')
    plt.savefig("imgs/"+ str(name)+".pdf", bbox_inches='tight', dpi=320)   
    plt.show() 
    
    
