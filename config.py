import os
from datetime import datetime
from pathlib import Path


def get_parameters(mode):

    # mode = 'reg' # available modes: 'reg', 'train', 'test'

    hyperparams = {'model_dims': (128, 128, 64), # Dimensiones de entrada al modelo
                   'new_z'     : [2, 2, 2],      # Nuevo tamano de zooms
                   'lr'        : 0.0001,         # Taza de aprendizaje
                   'epochs'    : 20,             # Numero de epocas
                   'batch_size': 1,              # Tama;o del batch
                   'crit'      : 'BCEDice',      # Fn de costo. Opciones: 'BCEDice', 'BCELog', 'CELoss', 'BCE', 'BCELogW'
                   'n_train'   : 100,            # Cabezas para entrenamiento. Total=295
                   'n_val'     : 15,             # "" Validacion. Total=37
                   'n_test'    : 15,             # "" Prueba. Total=37
                   'batchnorm' : False,          # Normalizacion de batch
                   'nclasses'  : 1,              # Numero de clases
                   'thres'     : 0.5,            # Umbral
                   'class_w'   : 5.,             # Peso ponderado de la clase
                   'crop'      : True,           # Recortar o no recortar slices sin fcd del volumen
    }

    labels = {'bgnd': 0, # Image background
              'NCR' : 1, # necrotic tumor core
              'ED'  : 2, # peritumoral edematous/invaded tissue
              'ET'  : 4, # GD-enhancing tumor
    }

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

    folder = './outs/Ex-' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    if mode == 'train':

        Path(folder).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(folder, 'val_imgs')).mkdir(parents=True, exist_ok=True)

        files = {'model' : os.path.join(folder, 'weights'),
                 'losses': os.path.join(folder, 'losses.csv'),
                 't_mets': os.path.join(folder, 'train_metrics.csv'),
                 'v_mets': os.path.join(folder, 'val_metrics.csv'),
                 'pics'  : os.path.join(folder, 'val_imgs', 'img'),
                 'params': os.path.join(folder, 'params.txt'),
                 'summary': os.path.join(folder, 'cnn_summary.txt'),
                 'log'    : os.path.join(folder, 'train.log')}

        return {'mode': mode,
                'data': datasets,
                'hyperparams': hyperparams,
                'files': files,
                'labels': labels,
        }

    elif mode == 'test':

        ex = './outs/imgs/weights-BCEDice-20_eps-100_heads-2023-07-03-_nobn(mejor)'
        mo = 'weights-BCEDice-20_eps-100_heads-2023-07-03-_nobn-e20.pth'

        test_folder = os.path.join(ex, 'test'+mo[:-4])
        Path(test_folder).mkdir(parents=True, exist_ok=True)
        img_folder = os.path.join(test_folder, 'imgs')
        Path(img_folder).mkdir(parents=True, exist_ok=True)

        PATH_TRAINED_MODEL = os.path.join(ex, mo) #'./outs/Ex/prueba.pth' # 'weights-bcedice-20_eps-100_heads-2023-03-10-_nobn.pth'
        PATH_TEST_METS = os.path.join(test_folder, mo+'-test_metrics.csv')#'./outs/Ex/test_metrics.csv'

        return {'mode'       : mode,
                'data'       : datasets,
                'hyperparams': hyperparams,
                'labels'     : labels,
                'weights'    : PATH_TRAINED_MODEL,
                'test_fn'    : PATH_TEST_METS,
                'img_folder' : img_folder,
        }

    elif mode == 'assess':

        ex = './outs/imgs/weights-BCEDice-20_eps-100_heads-2023-07-03-_nobn(mejor)'
        mo = 'weights-BCEDice-20_eps-100_heads-2023-07-03-_nobn-e20.pth'

        plots_folder = os.path.join(ex, 'plots-weights_'+mo[-7:-4])

        Path(plots_folder).mkdir(parents=True, exist_ok=True)

        train_losses = 'losses-BCEDice-20_eps-100_heads-2023-07-03-_nobn.csv'
        train_dices  = 't-dices-BCEDice-20_eps-100_heads-2023-07-03-_nobn.csv'
        val_dices = 'v-dices-BCEDice-20_eps-100_heads-2023-07-03-_nobn.csv'
        test_mets = 'weights-BCEDice-20_eps-100_heads-2023-07-03-_nobn-e20.pth-test_metrics.csv'

        files = {'train_Loss': os.path.join(ex, train_losses),
                 'train_Dice': os.path.join(ex, train_dices),
                 'val_Dice'  : os.path.join(ex, val_dices),
                 'test_mets' : os.path.join(ex, 'testweights-BCEDice-20_eps-100_heads-2023-07-03-_nobn-e20', test_mets)
        }

        return {'mode'     : mode,
                'labels'   : labels,
                'files'    : files,
                'plots'    : plots_folder,
        }

        # train_losses = './outs/Ex-2023-07-15-01-41-42/losses.csv'
        # train_metrics  = './outs/Ex-2023-07-15-01-41-42/t-accs.csv'
        # val_metrics  = './outs/Ex-2023-07-15-01-41-42/v-accs.csv'
        # test_metrics   = './outs/Ex-2023-07-15-01-41-42/test_metrics.csv'

        # files = {'train_Loss': train_losses,
        #          'train_mets': train_metrics,
        #          'val_mets'  : val_metrics,
        #          'test_mets' : test_metrics
        #         }

        # return {'mode'     : mode,
        #         'labels'   : labels,
        #         'files'    : files
        #         }
