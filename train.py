from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torchmetrics.functional import dice
from unet import Unet
from metrics import DiceLoss, dice_coeff, BCEDiceLoss, FocalLoss, TverskyLoss
from get_data import Unet2D_DS
from utils.plots import plot_overlays


def train(config):

    torch.cuda.empty_cache()

    start_time = datetime.now()
    print(f'\nHora de inicio: {start_time}')

    print(f"\nNum epochs = {config['hyperparams']['epochs']}, batch size = {config['hyperparams']['batch_size']}, \
        Learning rate = {config['hyperparams']['lr']}\n")

    print(f"Model file name = {config['files']['model_fn']}\n")

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
        ds_val, #val_mris, 
        batch_size=1 #config['batch_size'],
    )

    print(f'Tamano del dataset de entrenamiento: {len(ds_train)} slices')
    print(f'Tamano del dataset de validacion: {len(ds_val)} slices \n')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    unet = Unet(config['hyperparams']['nclasses'], depth=5, batchnorm=config['hyperparams']['batchnorm']).to(device, dtype=torch.double)
    #print(torch.cuda.memory_summary(device=device, abbreviated=False))

    criterion = {'CELoss' : nn.CrossEntropyLoss(),  # Cross entropy loss performs softmax by default
                 'BCELog' : nn.BCEWithLogitsLoss(), # BCEWithLogitsLoss performs sigmoid by default
                 'BCE'    : nn.BCELoss(),
                 'Dice'   : DiceLoss(),
                 'BCEDice': BCEDiceLoss(device=device),
                 'Focal'  : FocalLoss(),
                 'Tversky': TverskyLoss()
    }

    optimizer = Adam(unet.parameters(), lr=config['hyperparams']['lr'])

    best_loss = 1.0

    losses = []
    train_dices = []
    eval_dices  = []

    for epoch in tqdm(range(config['hyperparams']['epochs'])):  # loop over the dataset multiple times

        #torch.cuda.empty_cache()

        running_loss = 0.0
        running_dice = 0.0
        epoch_loss   = 0.0
        epoch_train_dice = 0
        
        print(f'\n\nEpoch {epoch + 1}\n')

        unet.train()
        for i, data in enumerate(train_dl, 0):

            inputs, labels = data
            #print(f'{torch.unique(labels)}, {torch.max(inputs)}')
            inputs = inputs.unsqueeze(1).to(device, dtype=torch.double)
            labels = labels.to(device, dtype=torch.double)

            outputs = unet(inputs)
            # print(f'{outputs.shape}, {torch.unique(labels)}')
            # print(f'En train = {torch.unique(outputs)}, {torch.max(outputs)}')

            loss = criterion['BCELog'](outputs.double(), labels.unsqueeze(1))#(outputs.double().squeeze(1), labels) # Utilizar esta linea para BCELoss o DiceLoss
            #loss = criterion(outputs, labels.long()) # Utilizar esta linea para Cross entropy loss (multiclase)

            '''Prueba'''
            probs_ = nn.Sigmoid()  # Sigmoid para biclase
            pval_  = probs_(outputs) 
            preds_ = torch.where(pval_>0.1, 1., 0.)
            batch_train_dice = dice_coeff(preds_, labels.unsqueeze(1))
            epoch_train_dice +=batch_train_dice.item()
            train_dices.append([epoch, i, batch_train_dice.item()])
            '''Fin prueba'''

            if (i+1) % 40 == 0: 
                print(f'Batch No. {(i+1)} loss = {loss.item():.3f}')
                print(f'Prueba = {torch.unique(preds_)}, max = {torch.max(preds_)}, dice = {batch_train_dice}')

            running_loss += loss.item()
            optimizer.zero_grad() # zero the parameter gradients
            loss.backward(retain_graph=True)#retain_graph=True
            #nn.utils.clip_grad_norm_(unet.parameters(), max_norm=2.0, norm_type=2)
            #nn.utils.clip_grad_value_(unet.parameters(), clip_value=1.0) # Gradient clipping
            optimizer.step()
            losses.append([epoch, i, loss.item()])
            
        epoch_loss = running_loss/(i + 1)  

        epoch_val_dice = 0   
        pvalmax = [] 

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

                batch_eval_dice = dice_coeff(preds, y.unsqueeze(1))#(preds.squeeze(1), y)
                #batch_eval_dice = dice(preds.squeeze(1), y.long(), ignore_index=0, zero_division=1) # version de torchmetrics de la metrica
                epoch_val_dice += batch_eval_dice.item()
                eval_dices.append([epoch, j, batch_eval_dice.item()])

                if (j+1) % 40 == 0: 
                    print(f'En eval = {torch.unique(preds)}, {torch.max(preds)}')
                    print(f'pval min = {pval.min():.3f}, pval max = {pval.max():.3f}')
                    print(f'Dice promedio hasta batch No. {j+1} = {epoch_val_dice/(j+1):.3f}')

                if (j+1) % 8 == 0:
                    if torch.any(y):
                        plot_overlays(x.squeeze(1), 
                                      y, 
                                      preds.squeeze(1), 
                                      mode='save', 
                                      fn=f"{config['files']['pics']}-epoca_{epoch + 1}-b{j}")


        epoch_val_dice = epoch_val_dice / (j+1) 

        if epoch == 0:
            best_loss = epoch_loss
            best_dice = epoch_val_dice

        if epoch_loss < best_loss:
            best_loss = epoch_loss

        print(f'\nEpoch loss = {epoch_loss:.3f}, Best loss = {best_loss:.3f}\n')
        print(f'Epoch dice (Training) = {epoch_train_dice / (i+1):.3f}\n')

        if epoch_val_dice > best_dice:
            best_dice = epoch_val_dice
            print(f'\nUpdated weights file!')
            torch.save(unet.state_dict(), config['files']['model_fn'])

        print(f'\nEpoch dice (Validation) = {epoch_val_dice:.3f}, Best dice = {best_dice:.3f}\n')

    df_loss = pd.DataFrame(losses, columns=['Epoca', 'Batch', 'Loss'])
    df_loss = df_loss.assign(id=df_loss.index.values)
    df_loss.to_csv(config['files']['losses_fn'])

    df_dice = pd.DataFrame(train_dices, columns=['Epoca', 'Batch', 'Dice'])
    df_dice = df_dice.assign(id=df_dice.index.values)
    df_dice.to_csv(config['files']['t_dice_fn'])

    df_dice = pd.DataFrame(eval_dices, columns=['Epoca', 'Batch', 'Dice'])
    df_dice = df_dice.assign(id=df_dice.index.values)
    df_dice.to_csv(config['files']['v_dice_fn'])

    print(f'\nFinished training. Total training time: {datetime.now() - start_time}\n')


