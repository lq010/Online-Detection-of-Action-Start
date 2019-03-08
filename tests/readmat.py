from scipy.io import loadmat
x = loadmat('res_seg_swin.mat')
predictions = x['seg_swin']
print(predictions.shape)