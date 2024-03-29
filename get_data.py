import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import nibabel as nib
import nibabel.processing


class Unet2D_DS(Dataset):

    def __init__(self, config, mode):

        self.config = config
        self.mode   = mode

        data_dir = ''
        n = 0

        if self.mode == 'train':
            data_dir = self.config['data']['train'] 
            n = self.config['hyperparams']['n_train']

        elif self.mode == 'val':
            data_dir = self.config['data']['val']
            n = self.config['hyperparams']['n_val']

        elif self.mode == 'test':
            data_dir = self.config['data']['test']
            n = self.config['hyperparams']['n_test']

        self.subjects = next(os.walk(data_dir))[1] # [2]: lists files; [1]: lists subdirectories; [0]: root

        self.L = []

        #n = config['n_heads']*config['model_dims'][2]

        for subject in self.subjects[:n]:
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

            l = preprocess(label_path, self.config, norm=True)
            for slice_ in range(self.config['hyperparams']['model_dims'][2]):
                if np.any(l[:, :, slice_]):
                    self.L.append([subject, slice_, mri_path, label_path])

        self.df = pd.DataFrame(self.L, columns=['Subject', 'Slice', 'Path MRI', 'Path Label'])
        self.df = self.df.assign(id=self.df.index.values).sample(frac=1)
        #print(f'dataframe: \n{self.df} \n')


    def __len__(self):

        return self.df.shape[0]


    def __getitem__(self, index):

        load_slice = self.df.at[index, 'Slice']

        mri    = preprocess(self.df.at[index, 'Path MRI'], self.config, norm=True)[:, :, load_slice]
        label_ = preprocess(self.df.at[index, 'Path Label'], self.config)[:, :, load_slice]
        label  = np.where((label_==self.config['labels']['NCR']) | (label_==self.config['labels']['ET']), 1., 0.)

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
        label  = np.where((label_==self.config['labels']['NCR']) | (label_==self.config['labels']['ET']), 1., 0.)

        return mri, label

def preprocess(path, config, norm=False):

    scan = nib.load(path)
    aff  = scan.affine
    vol  = scan.get_fdata() # np.int16(scan.get_fdata())
    #print(f'shape = {vol.shape}, path = {path}')

    # Remuestrea volumen y affine a un nuevo shape
    # new_zooms  = np.array(scan.header.get_zooms()) * config['new_z']
    # new_shape  = np.array(vol.shape) // config['new_z']
    
    new_affine = nibabel.affines.rescale_affine(aff, vol.shape, config['hyperparams']['new_z'], config['hyperparams']['model_dims']) # new_zooms, (128, 128, 64))#new_shape)
    scan       = nibabel.processing.conform(scan, config['hyperparams']['model_dims'], config['hyperparams']['new_z']) # (128, 128, 64), new_zooms)
    ni_img     = nib.Nifti1Image(scan.get_fdata(), new_affine)
    vol        = ni_img.get_fdata() #np.int16(ni_img.get_fdata())
    if norm:
        vol        = (vol - np.min(vol))/(np.max(vol) - np.min(vol))

    return vol


# def preprocess(path, config, norm=False):

#     scan = nib.load(path)
#     aff  = scan.affine

#     vol  = scan.get_fdata()

#     if 'Ras_msk' in path:

#         try:
#             vol = scan.get_fdata().squeeze(3)
#             scan = nib.Nifti1Image(vol, aff)
#         except:
#             vol = np.where(vol==0., 0., 1.)
#             scan = nib.Nifti1Image(vol, aff)


#     # Remuestrea volumen y affine a un nuevo shape

#     #new_zooms  = np.array(scan.header.get_zooms()) * config['new_z']
#     #new_shape  = np.array(vol.shape) // config['new_z']

#     new_affine = nibabel.affines.rescale_affine(aff, 
#                                                 vol.shape, 
#                                                 config['hyperparams']['new_z'], 
#                                                 config['hyperparams']['model_dims']
#     )

#     scan       = nibabel.processing.conform(scan, 
#                                             config['hyperparams']['model_dims'], 
#                                             config['hyperparams']['new_z']
#     )
     
#     ni_img     = nib.Nifti1Image(scan.get_fdata(), new_affine)
#     vol        = ni_img.get_fdata() 

#     if 'Ras_msk' in path:
#         vol = np.where(vol <= 0.1, 0., 1.)

#     if norm:
#         vol = (vol - np.min(vol))/(np.max(vol) - np.min(vol))

#     return vol