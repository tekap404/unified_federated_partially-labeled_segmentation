from matplotlib import pyplot as plt
import os
import numpy as np
import nibabel as nib

def save_npy(cfg, data, gt, outputs, name):
    os.makedirs(cfg.plot_path + '/' + name, exist_ok=True)

    new_image = (data[:,0]*255).astype(np.int16)
    nib.save(nib.Nifti1Image(new_image, np.eye(4)), cfg.plot_path + '/' + name + '/img.nii.gz')

    gt_kidney = gt[:,1].astype(np.int16)
    gt_liver = gt[:,2].astype(np.int16)
    gt_spleen = gt[:,3].astype(np.int16)
    gt_pancreas = gt[:,4].astype(np.int16)
    nib.save(nib.Nifti1Image(gt_kidney, np.eye(4)), cfg.plot_path + '/' + name + '/gt_kidney.nii.gz')
    nib.save(nib.Nifti1Image(gt_liver, np.eye(4)), cfg.plot_path + '/' + name + '/gt_liver.nii.gz')
    nib.save(nib.Nifti1Image(gt_spleen, np.eye(4)), cfg.plot_path + '/' + name + '/gt_spleen.nii.gz')
    nib.save(nib.Nifti1Image(gt_pancreas, np.eye(4)), cfg.plot_path + '/' + name + '/gt_pancreas.nii.gz')

    output_kidney = outputs[:,1].astype(np.int16)
    output_liver = outputs[:,2].astype(np.int16)
    output_spleen = outputs[:,3].astype(np.int16)
    output_pancreas = outputs[:,4].astype(np.int16)
    nib.save(nib.Nifti1Image(output_kidney, np.eye(4)), cfg.plot_path + '/' + name + '/pred_kidney.nii.gz')
    nib.save(nib.Nifti1Image(output_liver, np.eye(4)), cfg.plot_path + '/' + name + '/pred_liver.nii.gz')
    nib.save(nib.Nifti1Image(output_spleen, np.eye(4)), cfg.plot_path + '/' + name + '/pred_spleen.nii.gz')
    nib.save(nib.Nifti1Image(output_pancreas, np.eye(4)), cfg.plot_path + '/' + name + '/pred_pancreas.nii.gz')

def vis_data_label(cfg, data, gt, outputs, name):
    os.makedirs(cfg.plot_path + '/' + name, exist_ok=True)
    for i in range(0, gt.shape[0]):
        plt.figure(dpi=100)

        plt.subplot(121)
        plt.imshow(data[i,0,:,:], cmap=plt.cm.gray)
        plt.imshow(gt[i,1,:,:], cmap=plt.cm.Purples, alpha=0.2)
        plt.imshow(gt[i,2,:,:], cmap=plt.cm.Greens, alpha=0.2)
        plt.imshow(gt[i,3,:,:], cmap=plt.cm.Blues, alpha=0.2)
        plt.imshow(gt[i,4,:,:], cmap=plt.cm.Reds, alpha=0.2)
        plt.axis('OFF')
        plt.title('ground truth')

        plt.subplot(122)
        plt.imshow(data[i,0,:,:], cmap=plt.cm.gray)
        plt.imshow(outputs[i,1,:,:], cmap=plt.cm.Purples, alpha=0.2)
        plt.imshow(outputs[i,2,:,:], cmap=plt.cm.Greens, alpha=0.2)
        plt.imshow(outputs[i,3,:,:], cmap=plt.cm.Blues, alpha=0.2)
        plt.imshow(outputs[i,4,:,:], cmap=plt.cm.Reds, alpha=0.2)
        plt.axis('OFF')
        plt.title('pred')
        
        plt.show()
        plt.savefig(cfg.plot_path + '/' + name +'/img' + str(i) + '.jpg')
        plt.close()