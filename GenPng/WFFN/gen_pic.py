import pandas as pd
import matplotlib.pyplot as plt

train_log = pd.read_csv("WFFN.train")
test_log = pd.read_csv("WFFN.test")

_,ax1 = plt.subplots()
ax1.set_title("WFFN")
ax1.plot(train_log["NumIters"], train_log["train_loss"], alpha=0.5)
ax1.plot(test_log["NumIters"], test_log["test_loss"], 'g')
#ax1.plot(test_log["NumIters"],test_log["scale1_loss"])
ax1.set_xlabel('iteration')
ax1.set_ylabel('loss')
plt.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(train_log["NumIters"], train_log["train_acc"], 'y')
ax2.plot(test_log["NumIters"], test_log["test_acc"], 'r')
ax2.set_ylabel('accuracy')
plt.legend(loc='upper right')

plt.show()
