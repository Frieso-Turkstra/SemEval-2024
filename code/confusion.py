import json
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


test_data = [json.loads(line) for line in open("../SubtaskA/subtaskA_dev_multilingual.jsonl", 'r')]
Y_test = [obj['label'] for obj in test_data]
labels = np.unique(Y_test)
pred_data = [json.loads(line) for line in open("pred_A_multi.jsonl", "r")]
Y_pred = [obj['label'] for obj in pred_data]
cm = confusion_matrix(Y_test, Y_pred, labels=labels)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
display.plot()
plt.show()
