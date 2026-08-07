import numpy as np
b1 = np.fromfile('weights/b1.bin', dtype=np.float32)
b2 = np.fromfile('weights/b2.bin', dtype=np.float32)
W1 = np.fromfile('weights/W1.bin', dtype=np.float32)
W2 = np.fromfile('weights/W2.bin', dtype=np.float32)
test_labels = np.fromfile('weights/test_labels.bin', dtype=np.int32)
pytorch_data = np.fromfile('weights/test_preds_pytorch.bin', dtype=np.int32)

print("b1 size: ", b1[0])
print("b2 size: ", b2[0])
print("W1 size: ", (W1[0]))
print("W2 size: ", W2[0])

print(len(W1))
print(test_labels)