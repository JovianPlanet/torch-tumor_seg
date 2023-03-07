import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import nibabel as nib
import nibabel.processing


class Unet2D_DS(Dataset):

    def __init__(self, config):

        self.config = config

        data_dir = ''

        if self.config['mode'] == 'train':
            data_dir = self.config['brats_train'] # [2]: lists files; [1]: lists subdirectories; [0]: root
        elif self.config['mode'] == 'test':
            data_dir = self.config['brats_val'] # [2]: lists files; [1]: lists subdirectories; [0]: root

        self.subjects = next(os.walk(data_dir))[1] # [2]: lists files; [1]: lists subdirectories; [0]: root

        self.L = []

        n = config['n_heads']*config['model_dims'][2]

        for subject in self.subjects:
            if '355' in subject: continue
            #print(f'\nsujeto: {subject}')
            files = next(os.walk(os.path.join(data_dir, subject)))[2]
            for file_ in files:
                if 't1.nii' in file_:
                    #print(f'\tfile_: {file_}')
                    mri_path = os.path.join(data_dir, subject, file_)
                if 'seg.nii' in file_:
                    #print(f'\tfile_: {file_}')
                    label_path = os.path.join(data_dir, subject, file_)

            for slice_ in range(self.config['model_dims'][2]):
                self.L.append([subject, slice_, mri_path, label_path])

        self.df = pd.DataFrame(self.L[:n], columns=['Subject', 'Slice', 'Path MRI', 'Path Label'])
        self.df = self.df.assign(id=self.df.index.values).sample(frac=1)
        print(f'dataframe: \n{self.df} \n')


    def __len__(self):

        return self.df.shape[0]


    def __getitem__(self, index):

        load_slice = self.df.at[index, 'Slice']

        #mri    = np.int16(nib.load(self.df.at[index, 'Path MRI']).get_data())[:, :, load_slice]
        #label_ = np.int16(nib.load(self.df.at[index, 'Path Label']).get_data())[:, :, load_slice]

        mri    = preprocess(self.df.at[index, 'Path MRI'], self.config, norm=False)[:, :, load_slice]
        label_ = preprocess(self.df.at[index, 'Path Label'], self.config)[:, :, load_slice]
        label  = np.where((label_==self.config['labels']['NCR']) | (label_==self.config['labels']['ET']), 1, 0)

        return mri, label


class Unet3D_DS(Dataset):

    def __init__(self, config):


        self.config = config

        self.subjects = next(os.walk(self.config['brats_train']))[1] # [2]: lists files; [1]: lists subdirectories; [0]: ?

        self.L = []

        for subject in self.subjects:
            if '355' in subject: continue
            #print(f'\nsujeto: {subject}')
            files = next(os.walk(os.path.join(self.config['brats_train'], subject)))[2]
            for file_ in files:
                if 't1.nii' in file_:
                    #print(f'\tfile_: {file_}')
                    mri_path = os.path.join(self.config['brats_train'], subject, file_)
                if 'seg.nii' in file_:
                    #print(f'\tfile_: {file_}')
                    label_path = os.path.join(self.config['brats_train'], subject, file_)

            self.L.append([subject, mri_path, label_path])

        self.df = pd.DataFrame(self.L, columns=['Subject', 'Path MRI', 'Path Label'])
        self.df = self.df.assign(id=self.df.index.values).sample(frac=1)
        print(f'dataframe: \n{self.df} \n')


    def __len__(self):

        return self.df.shape[0]


    def __getitem__(self, index):

        mri    = np.int16(nib.load(self.df.at[index, 'Path MRI']).get_data())
        label_ = np.int16(nib.load(self.df.at[index, 'Path Label']).get_data())
        label  = np.where((label_==self.config['labels']['NCR']) | (label_==self.config['labels']['ET']), 1, 0)

        return mri, label

def preprocess(path, config, norm=False):

    scan = nib.load(path)
    aff  = scan.affine
    vol  = np.int16(scan.get_fdata())
    #print(f'shape = {vol.shape}, path = {path}')

    # Resamplea volumen y affine a un nuevo shape
    new_zooms  = np.array(scan.header.get_zooms()) * config['new_z']
    new_shape  = np.array(vol.shape) // config['new_z']
    new_affine = nibabel.affines.rescale_affine(aff, vol.shape, new_zooms, (128, 128, 64))#new_shape)
    scan       = nibabel.processing.conform(scan, (128, 128, 64), new_zooms)
    ni_img     = nib.Nifti1Image(scan.get_fdata(), new_affine)
    vol        = np.int16(ni_img.get_fdata())
    if norm:
        vol        = (vol - np.min(vol))/(np.max(vol) - np.min(vol))

    return vol