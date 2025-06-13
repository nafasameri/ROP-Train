import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Extract epoch numbers
epochs = np.arange(1, 201)

# Training Loss data
training_loss = [
    0.6060, 0.4112, 0.4105, 0.3113, 0.2457, 0.2481, 0.1906, 0.1359, 0.1014, 0.1356,
    0.1285, 0.1849, 0.0949, 0.0746, 0.0483, 0.0476, 0.0744, 0.0735, 0.1052, 0.1019,
    0.0867, 0.1012, 0.0481, 0.0362, 0.0618, 0.0780, 0.0785, 0.0790, 0.0455, 0.0296,
    0.0523, 0.0604, 0.1013, 0.1375, 0.0645, 0.0423, 0.0216, 0.0391, 0.0796, 0.1403,
    0.0599, 0.0261, 0.0268, 0.0236, 0.0207, 0.0233, 0.0350, 0.0806, 0.0886, 0.0382,
    0.0202, 0.0157, 0.0161, 0.0150, 0.0145, 0.0150, 0.0160, 0.0142, 0.0144, 0.0139,
    0.0147, 0.0138, 0.0137, 0.0148, 0.0134, 0.0137, 0.0133, 0.0134, 0.0151, 0.0143,
    0.0133, 0.0141, 0.0143, 0.0136, 0.0139, 0.0131, 0.0136, 0.0139, 0.0133, 0.0133,
    0.0135, 0.0141, 0.0135, 0.0133, 0.0135, 0.0140, 0.0131, 0.0139, 0.0135, 0.0133,
    0.0140, 0.0133, 0.0132, 0.0133, 0.0133, 0.0140, 0.0132, 0.0132, 0.0134, 0.0133,
    0.0132, 0.0147, 0.0132, 0.0133, 0.0132, 0.0132, 0.0138, 0.0132, 0.0132, 0.0131,
    0.0134, 0.1970, 0.7166, 0.6688, 0.5680, 0.4720, 0.3920, 0.3250, 0.2097, 0.1630, 0.1882,
    0.1381, 0.0838, 0.0871, 0.0705, 0.0917, 0.0715, 0.0423, 0.0342, 0.0276, 0.0306,
    0.0455, 0.0466, 0.0452, 0.0432, 0.0493, 0.0988, 0.0727, 0.0586, 0.0591, 0.0417,
    0.0298, 0.0233, 0.0203, 0.0271, 0.0484, 0.0437, 0.0388, 0.0547, 0.0775, 0.0420,
    0.0250, 0.0382, 0.0341, 0.0333, 0.0304, 0.0217, 0.0189, 0.0199, 0.0346, 0.0385,
    0.0615, 0.0551, 0.0306, 0.0436, 0.0964, 0.0887, 0.0540, 0.0299, 0.0367, 0.0197,
    0.0144, 0.0153, 0.0167, 0.0151, 0.0150, 0.0143, 0.0139, 0.0137, 0.0368, 0.1931,
    0.0606, 0.0455, 0.0268, 0.0274, 0.0207, 0.0161, 0.0177, 0.0149, 0.0146, 0.0146,
    0.0136, 0.0137, 0.0134, 0.0137, 0.0140, 0.0139, 0.0144, 0.0135, 0.0136
]

# Validation Loss data
validation_loss = [
    0.7058, 0.6572, 0.4080, 0.4225, 0.5574, 0.3534, 0.4955, 0.3824, 2.0611, 0.6037,
    1.6979, 0.3378, 0.4754, 0.3126, 0.4615, 0.4970, 0.6896, 0.8374, 0.8591, 0.3521,
    0.9857, 0.4211, 0.3180, 0.5531, 0.6337, 0.7673, 0.6719, 0.6833, 0.2844, 0.3226,
    0.6773, 0.8511, 0.4106, 1.0116, 0.8173, 0.5265, 0.3911, 0.4397, 0.5325, 0.5368,
    0.6272, 0.4654, 0.5952, 0.5368, 1.0616, 0.5107, 0.6411, 0.5638, 0.5823, 0.4128,
    0.4171, 0.4265, 0.4585, 0.4263, 0.4939, 0.4674, 0.5165, 0.4875, 0.4711, 0.4594,
    0.4711, 0.4760, 0.4592, 0.5205, 0.5326, 0.5456, 0.5365, 0.5159, 0.4721, 0.4855,
    0.5072, 0.4753, 0.6028, 0.5591, 0.5573, 0.5566, 0.5290, 0.5563, 0.5070, 0.5358,
    0.5216, 0.5411, 0.5412, 0.5261, 0.5326, 0.5219, 0.5292, 0.5279, 0.5206, 0.6037,
    0.5514, 0.5638, 0.5255, 0.5521, 0.5445, 0.5238, 0.5391, 0.5360, 0.5386, 0.5319,
    0.5300, 0.5476, 0.5152, 0.5275, 0.5376, 0.5324, 0.5521, 0.5334, 0.5208, 0.5063,
    0.5083, 6.2468, 93.4787, 1.1194, 0.5880, 0.5332, 1.7232, 1.1066, 0.5474, 0.5527, 0.4021,
    0.4587, 0.4649, 0.5069, 0.4138, 0.5003, 0.6953, 0.6246, 0.5906, 0.5990, 0.6785,
    0.7647, 0.5753, 0.5940, 0.5254, 0.9694, 0.6828, 0.8610, 0.5016, 0.7872, 0.5870,
    0.5092, 0.4921, 0.6193, 0.6328, 0.6975, 0.5965, 0.7722, 0.4536, 0.7106, 0.4437,
    0.4858, 0.6588, 0.7991, 1.3820, 0.4492, 0.4442, 0.6028, 0.4806, 0.5325, 0.8121,
    0.8499, 0.9649, 0.5832, 0.5077, 1.3143, 0.6300, 2.1389, 0.5308, 0.5377, 0.5254,
    0.4923, 0.5044, 0.5101, 0.5507, 0.5391, 0.5151, 0.5118, 0.5164, 1.1082, 0.5255,
    0.8857, 0.5043, 0.4376, 0.4523, 0.4050, 0.3472, 0.3659, 0.3744, 0.3297, 0.3752,
    0.3616, 0.3695, 0.3571, 0.3659, 0.3819, 0.3672, 0.3813, 0.3917, 0.4014
]

# Create figure with two subplots
plt.figure(figsize=(12, 6))

# Plot Training Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, training_loss, 'b-', label='Training Loss')
plt.title('Training Loss over 200 Epochs', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Highlight anomalies in training loss
anomaly_epochs = [112, 113, 180, 181]
for epoch in anomaly_epochs:
    plt.scatter(epoch, training_loss[epoch-1], color='red', s=100, zorder=5)
    plt.text(epoch, training_loss[epoch-1], f"Epoch {epoch}",
             ha='right', va='bottom', color='red', fontsize=10)

# Plot Validation Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, validation_loss, 'r-', label='Validation Loss')
plt.title('Validation Loss over 200 Epochs', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Highlight anomalies in validation loss
anomaly_epochs_val = [9, 11, 45, 112, 113, 168]
for epoch in anomaly_epochs_val:
    plt.scatter(epoch, validation_loss[epoch-1], color='red', s=100, zorder=5)
    plt.text(epoch, validation_loss[epoch-1], f"Epoch {epoch}",
             ha='right', va='bottom', color='red', fontsize=10)

# plt.tight_layout()
# plt.show()

# Combined plot to see both curves together
# plt.figure(figsize=(12, 6))
# plt.plot(epochs, training_loss, 'b-', label='Training Loss')
# plt.plot(epochs, validation_loss, 'r-', label='Validation Loss')
# plt.title('Training and Validation Loss over 200 Epochs', fontsize=14)
# plt.xlabel('Epoch', fontsize=12)
# plt.ylabel('Loss', fontsize=12)
# plt.grid(True, alpha=0.3)
# plt.legend(fontsize=12)
#
# # Highlight major anomalies
# # major_anomalies = [112, 113]
# # for epoch in major_anomalies:
# #     plt.scatter(epoch, training_loss[epoch-1], color='purple', s=100, zorder=5)
# #     plt.scatter(epoch, validation_loss[epoch-1], color='purple', s=100, zorder=5)
# #     # plt.text(epoch, max(training_loss[epoch-1]), f"Epoch {epoch}", ha='right', va='bottom', color='purple', fontsize=10)
#
# # plt.tight_layout()
# # plt.show()
plt.savefig('loss.png')
