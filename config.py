import os
import datetime

def get_parameters(mode):

    # mode = 'reg' # available modes: 'reg', 'train', 'test'

    model_dims = (128, 128, 64)
    lr         = 0.001
    epochs     = 20
    batch_size = 8
    new_z      = [2, 2, 2]
    n_heads    = 100 #368 Total cabezas disponibles entrenamiento: 295
    n_train    = 295
    n_val      = 37
    n_test     = 37

    labels = {'bgnd': 0, # Image background
              'NCR' : 1, # necrotic tumor core
              'ED'  : 2, # peritumoral edematous/invaded tissue
              'ET'  : 4, # GD-enhancing tumor
    }

    model_fn  = 'weights-bcedice-'+str(epochs)+'_eps-'+str(n_train)+'_heads-'+str(datetime.date.today())+'.pth'
    losses_fn = './outs/losses-bcedice-'+str(epochs)+'_eps-'+str(n_train)+'_heads-'+str(datetime.date.today())+'.csv'
    dices_fn  = './outs/dices-bcedice-'+str(epochs)+'_eps-'+str(n_train)+'_heads-'+str(datetime.date.today())+'.csv'

    brats_train = os.path.join('/home',
                              'davidjm',
                              'Downloads',
                              'BraTS-dataset',
                              'train',
                              #'MICCAI_BraTS2020_TrainingData'
    )

    brats_val = os.path.join('/home',
                             'davidjm',
                             'Downloads',
                             'BraTS-dataset',
                             'val',
    )

    brats_test = os.path.join('/home',
                             'davidjm',
                             'Downloads',
                             'BraTS-dataset',
                             'test',
    )

    datasets = {'train': brats_train, 'val': brats_val, 'test': brats_test}


    if mode == 'train':

        res_path = os.path.join('/media',
                                'davidjm',
                                'Disco_Compartido',
                                'david',
                                'clsfr-tumors',
                                'results'
        )

        return {'mode'        : mode,
                'data'        : datasets,
                'model_dims'  : model_dims,
                'lr'          : lr,
                'epochs'      : epochs,
                'batch_size'  : batch_size,
                'new_z'       : new_z,
                'n_heads'     : n_heads,
                'n_train'     : n_train,
                'n_val'       : n_val,
                'model_fn'    : model_fn,
                'losses_fn'   : losses_fn,
                'dices_fn'    : dices_fn,
                'res_path'    : res_path, 
                'labels'      : labels,
        }

    elif mode == 'test':

        threshold = 0.5

        return {'mode'        : mode,
                'data'        : datasets,
                'model_dims'  : model_dims,
                'lr'          : lr,
                'epochs'      : epochs,
                'batch_size'  : batch_size,
                'new_z'       : new_z,
                'n_heads'     : n_heads,
                'n_test'      : n_test,
                'thres'       : threshold,
                'model_fn'    : model_fn,
                'labels'      : labels,
        }
