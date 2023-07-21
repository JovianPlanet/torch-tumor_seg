from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from torchmetrics.functional import dice
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, Dice, BinaryRecall, BinaryPrecision, BinaryJaccardIndex
from torchinfo import summary

from unet import Unet
from metrics import DiceLoss, dice_coeff, BCEDiceLoss, FocalLoss, TverskyLoss
from get_data import Unet2D_DS
from utils.plots import plot_overlays
from utils.write_params import conf_txt, summary_txt


def train(config):

    torch.cuda.empty_cache()

    start_time = datetime.now()

    print(f'\nHora de inicio: {start_time}')
    print(f"\nEpocas = {config['hyperparams']['epochs']}, batch size = {config['hyperparams']['batch_size']}")
    print(f"Learning rate = {config['hyperparams']['lr']}\n")
    print(f"Nombre de archivo del modelo: {config['files']['model']}\n")


    # Crear datasets #
    ds_train = Unet2D_DS(config, 'train')
    ds_val   = Unet2D_DS(config, 'val')

    # train_size = int(0.8 * len(ds))
    # test_size  = len(ds) - train_size
    # train_mris, val_mris = random_split(ds, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(
        ds_train, #train_mris, 
        batch_size=config['hyperparams']['batch_size'],
        shuffle=True,
    )

    val_dl = DataLoader(
        ds_val,  
        batch_size=1,
        #shuffle=True,
    )

    print(f'Tamano del dataset de entrenamiento: {len(ds_train)} slices')
    print(f'Tamano del dataset de validacion: {len(ds_val)} slices \n')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    unet = Unet(config['hyperparams']['nclasses'], depth=5, batchnorm=config['hyperparams']['batchnorm']).to(device, dtype=torch.double)
    #print(torch.cuda.memory_summary(device=device, abbreviated=False))

    summary(unet)

    criterion = {'CELoss' : nn.CrossEntropyLoss(),  # Cross entropy loss performs softmax by default
                 'BCELog' : nn.BCEWithLogitsLoss(), # BCEWithLogitsLoss performs sigmoid by default
                 'BCE'    : nn.BCELoss(),
                 'Dice'   : DiceLoss(),
                 'BCEDice': BCEDiceLoss(device=device),
                 'Focal'  : FocalLoss(),
                 'Tversky': TverskyLoss()
    }

    optimizer = Adam(unet.parameters(), lr=config['hyperparams']['lr'])
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=4)

    t_acc = BinaryAccuracy(threshold=0.1).to(device, dtype=torch.double)
    t_dic = Dice(threshold=0.1).to(device, dtype=torch.double) # zero_division=1
    t_f1s = BinaryF1Score(threshold=0.1).to(device, dtype=torch.double)
    t_rec = BinaryRecall(threshold=0.1).to(device, dtype=torch.double)
    t_pre = BinaryPrecision(threshold=0.1).to(device, dtype=torch.double)
    t_jac = BinaryJaccardIndex(threshold=0.1).to(device, dtype=torch.double)

    v_acc = BinaryAccuracy(threshold=0.1).to(device, dtype=torch.double)
    v_dic = Dice(threshold=0.1).to(device, dtype=torch.double) # zero_division=1
    v_f1s = BinaryF1Score(threshold=0.1).to(device, dtype=torch.double)
    v_rec = BinaryRecall(threshold=0.1).to(device, dtype=torch.double)
    v_pre = BinaryPrecision(threshold=0.1).to(device, dtype=torch.double)
    v_jac = BinaryJaccardIndex(threshold=0.1).to(device, dtype=torch.double)

    conf_txt(config)
    summary_txt(config, str(summary(unet)))

    best_loss = 1.0
    best_ep_loss = 0
    best_ep_dice = 0
    best_dice = None
    best_acc = None

    losses = []
    train_metrics = []
    val_metrics = []

    for epoch in tqdm(range(config['hyperparams']['epochs'])):  # loop over the dataset multiple times

        #torch.cuda.empty_cache()

        running_loss = 0.0
        running_dice = 0.0
        epoch_loss   = 0.0
        ep_tr_dice = 0
        
        print(f'\n\nEpoch {epoch + 1}\n')

        unet.train()
        for i, data in enumerate(train_dl, 0):

            inputs, labels = data
            inputs = inputs.unsqueeze(1).to(device, dtype=torch.double)
            labels = labels.to(device, dtype=torch.double)

            outputs = unet(inputs)

            loss = criterion[config['hyperparams']['crit']](outputs.double(), labels.unsqueeze(1))#(outputs.double().squeeze(1), labels) # Utilizar esta linea para BCELoss o DiceLoss
            #loss = criterion(outputs, labels.long()) # Utilizar esta linea para Cross entropy loss (multiclase)

            '''Metricas'''
            probs_ = nn.Sigmoid()  # Sigmoid para biclase
            pval_  = probs_(outputs) 
            preds_ = torch.where(pval_>0.1, 1., 0.)
            # Dice coefficient
            ba_tr_dice = dice_coeff(preds_, labels.unsqueeze(1))
            ep_tr_dice += ba_tr_dice.item()
            # Torchmetrics
            t_dic.update(outputs, labels.unsqueeze(1).long())
            train_metrics.append([epoch, 
                                  i, 
                                  t_acc.forward(outputs, labels.unsqueeze(1)).item(),
                                  ba_tr_dice.item(), #t_dic.forward(outputs, labels.unsqueeze(1).long()).item(),
                                  t_f1s.forward(outputs, labels.unsqueeze(1)).item(),
                                  t_rec.forward(outputs, labels.unsqueeze(1)).item(),
                                  t_pre.forward(outputs, labels.unsqueeze(1)).item(),
                                  t_jac.forward(outputs, labels.unsqueeze(1)).item(),
                                  ]
            ) 
            '''Fin metricas'''

            running_loss += loss.item()
            optimizer.zero_grad() # zero the parameter gradients
            loss.backward(retain_graph=True)#retain_graph=True
            #nn.utils.clip_grad_norm_(unet.parameters(), max_norm=2.0, norm_type=2)
            #nn.utils.clip_grad_value_(unet.parameters(), clip_value=1.0) # Gradient clipping
            optimizer.step()
            losses.append([epoch, i, loss.item()])

            if (i+1) % 512 == 0: 
                print(f'\nMetricas promedio (Entrenamiento). Batch No. {i+1}')
                print(f'Loss = {running_loss/(i+1):.3f}')
                print(f'Dice = {ep_tr_dice/(i+1):.3f}')

                print(f'Torchmetrics')
                print(f'Accuracy  = {t_acc.compute():.3f}')
                print(f'Dice      = {t_dic.compute():.3f}') 
                print(f'F1 Score  = {t_f1s.compute():.3f}')
                print(f'Recall    = {t_rec.compute():.3f}')
                print(f'Precision = {t_pre.compute():.3f}')
                print(f'Jaccard   = {t_jac.compute():.3f}')

        before_lr = optimizer.param_groups[0]["lr"]
        if (epoch + 1) % 5:
            scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]

        epoch_loss = running_loss/(i + 1)  

        ep_val_dice = 0 
        ep_val_acc = 0  

        with torch.no_grad():
            unet.eval()
            print(f'\nValidacion\n')
            for j, testdata in enumerate(val_dl):
                x, y = testdata
                x = x.unsqueeze(1).to(device, dtype=torch.double)
                y = y.to(device, dtype=torch.double)

                outs  = unet(x)
                #probs = nn.Softmax(dim=1) # Softmax para multiclase
                probs = nn.Sigmoid()  # Sigmoid para biclase
                pval  = probs(outs) 
                preds = torch.where(pval>0.1, 1., 0.)
                #preds = torch.argmax(pval, dim=1)

                # Dice coefficient
                batch_val_dice = dice_coeff(preds, y.unsqueeze(1))#(preds.squeeze(1), y)
                ep_val_dice += batch_val_dice.item()
                #Torchmetrics
                v_dic.update(outs, y.unsqueeze(1).long())
                val_metrics.append([epoch, 
                                      j, 
                                      v_acc.forward(outs, y.unsqueeze(1)).item(),
                                      batch_val_dice.item(), #v_dic.forward(outs, y.unsqueeze(1).long()).item(),
                                      v_f1s.forward(outs, y.unsqueeze(1)).item(),
                                      v_rec.forward(outs, y.unsqueeze(1)).item(),
                                      v_pre.forward(outs, y.unsqueeze(1)).item(),
                                      v_jac.forward(outs, y.unsqueeze(1)).item(),
                                      ]
                ) 

                if (j+1) % 80 == 0: 

                    print(f'Metricas promedio (Validacion). Batch No. {j+1}')
                    print(f'Dice     = {ep_val_dice/(j+1):.3f}')

                    print(f'Torchmetrics')
                    print(f'Accuracy  = {v_acc.compute():.3f}')
                    print(f'Dice      = {v_dic.compute():.3f}') 
                    print(f'F1 Score  = {v_f1s.compute():.3f}')
                    print(f'Recall    = {v_rec.compute():.3f}')
                    print(f'Precision = {v_pre.compute():.3f}')
                    print(f'Jaccard   = {v_jac.compute():.3f}\n')

                if (j+1) % 16 == 0:
                    if torch.any(y):
                        plot_overlays(x.squeeze(1), 
                                      y, 
                                      preds.squeeze(1), 
                                      mode='save', 
                                      fn=f"{config['files']['pics']}-ep_{epoch + 1}-b{j}.pdf")


        ep_val_dice = ep_val_dice / (j+1) 
        ep_val_acc  = v_acc.compute()

        if epoch == 0:
            best_loss = epoch_loss
            best_ep_loss = epoch + 1
            best_dice = ep_val_dice
            best_ep_dice = epoch + 1
            best_acc = ep_val_acc

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_ep_loss = epoch + 1

        if ep_val_dice > best_dice:
            best_dice = ep_val_dice
            best_ep_dice = epoch + 1
            #print(f'\nUpdated weights file!')
        torch.save(unet.state_dict(), config['files']['model']+f'-e{epoch+1}.pth')

        if ep_val_acc > best_acc:
            best_acc = ep_val_acc

        print(f'\nMetricas totales. Epoca {epoch+1} (Entrenamiento)')
        print(f'Loss     = {epoch_loss:.3f}, Best loss = {best_loss:.3f} (epoca {best_ep_loss})')
        print(f'Dice     = {ep_tr_dice / (i+1):.3f}')

        print(f'Torchmetrics:')
        print(f'Accuracy  = {t_acc.compute():.3f}')
        print(f'Dice      = {t_dic.compute():.3f}')
        print(f'F1 Score  = {t_f1s.compute():.3f}')
        print(f'Recall    = {t_rec.compute():.3f}')
        print(f'Precision = {t_pre.compute():.3f}')
        print(f'Jaccard   = {t_jac.compute():.3f}')

        print(f'\nMetricas totales. Epoca {epoch+1} (Validacion):')
        print(f'Dice     = {ep_val_dice:.3f}, Best dice = {best_dice:.3f} (epoca {best_ep_dice})')
        print(f'Accuracy = {ep_val_acc:.3f}, Best accuracy = {best_acc:.3f}')
        
        print(f'Torchmetrics:')
        print(f'Accuracy  = {v_acc.compute():.3f}')
        print(f'Dice      = {v_dic.compute():.3f}')
        print(f'F1 Score  = {v_f1s.compute():.3f}')
        print(f'Recall    = {v_rec.compute():.3f}')
        print(f'Precision = {v_pre.compute():.3f}')
        print(f'Jaccard   = {v_jac.compute():.3f}\n')

        print(f'lr = {before_lr} -> {after_lr}\n')

        t_acc.reset()
        t_dic.reset()
        t_f1s.reset()
        t_rec.reset()
        t_pre.reset()
        t_jac.reset()

        v_acc.reset()
        v_dic.reset()
        v_f1s.reset()
        v_rec.reset()
        v_pre.reset() 
        v_jac.reset() 


    df_loss = pd.DataFrame(losses, columns=['Epoca', 'Batch', 'Loss'])
    df_loss = df_loss.assign(id=df_loss.index.values)
    df_loss.to_csv(config['files']['losses'])

    df_train = pd.DataFrame(train_metrics, columns=['Epoca', 'Batch', 'Accuracy', 'Dice', 'F1Score', 'Recall', 'Precision', 'Jaccard'])
    df_train = df_train.assign(id=df_train.index.values)
    df_train.to_csv(config['files']['t_mets'])

    df_val = pd.DataFrame(val_metrics, columns=['Epoca', 'Batch', 'Accuracy', 'Dice', 'F1Score', 'Recall', 'Precision', 'Jaccard'])
    df_val = df_val.assign(id=df_val.index.values)
    df_val.to_csv(config['files']['v_mets'])

    print(f'\nFinished training. Total training time: {datetime.now() - start_time}\n')


