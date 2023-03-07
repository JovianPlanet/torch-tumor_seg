import os
import datetime

def get_parameters(mode):

    # mode = 'reg' # available modes: 'reg', 'train', 'test'

    model_dims = (128, 128, 64)
    lr         = 0.001
    epochs     = 20
    batch_size = 8
    new_z      = [2, 2, 2]
    n_heads    = 100 #368
    # n_train    = 10
    # n_val      = 15
    # n_test     = 15

    labels = {'bgnd': 0, # Image background
              'NCR' : 1, # necrotic tumor core
              'ED'  : 2, # peritumoral edematous/invaded tissue
              'ET'  : 4, # GD-enhancing tumor
    }

    model_fn  = 'weights-bcedice-'+str(epochs)+'_eps-'+str(n_heads)+'_heads-'+str(datetime.date.today())+'.pth'
    losses_fn = './outs/losses-bcedice-'+str(epochs)+'_eps-'+str(n_heads)+'_heads-'+str(datetime.date.today())+'.csv'
    dices_fn  = './outs/dices-bcedice-'+str(epochs)+'_eps-'+str(n_heads)+'_heads-'+str(datetime.date.today())+'.csv'

    brats_train = os.path.join('/home',
                              'davidjm',
                              'Downloads',
                              'BraTS-dataset',
                              'BraTS2020_TrainingData',
                              'MICCAI_BraTS2020_TrainingData'
    )

    brats_val = os.path.join('/home',
                             'davidjm',
                             'Downloads',
                             'BraTS-dataset',
                             'test',
    )


    if mode == 'train':

        res_path = os.path.join('/media',
                                'davidjm',
                                'Disco_Compartido',
                                'david',
                                'clsfr-tumors',
                                'results'
        )

        return {'mode'          : mode,
                'brats_train'   : brats_train,
                'brats_val'     : brats_val,
                'model_dims'    : model_dims,
                'lr'            : lr,
                'epochs'        : epochs,
                'batch_size'    : batch_size,
                'new_z'         : new_z,
                'n_heads'       : n_heads,
                # 'n_train'       : n_train,
                # 'n_val'         : n_val,
                # 'n_test'        : n_test,
                'model_fn'      : model_fn,
                'losses_fn'     : losses_fn,
                'dices_fn'      : dices_fn,
                'res_path'      : res_path, 
                'labels'        : labels,
        }

    elif mode == 'test':

        threshold = 0.5

        return {'mode'               : mode,
                'brats_train'        : brats_train,
                'brats_val'          : brats_val,
                'model_dims'         : model_dims,
                'lr'                 : lr,
                'epochs'             : epochs,
                'batch_size'         : batch_size,
                'new_z'              : new_z,
                'n_heads'            : n_heads,
                # 'n_train'            : n_train,
                # 'n_val'              : n_val,
                # 'n_test'             : n_test,
                'thres'              : threshold,
                'model_fn'           : model_fn,
                'labels'             : labels,
        }
