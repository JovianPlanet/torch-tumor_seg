import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from unet import Unet
from metrics import DiceLoss, dice_coeff, BCEDiceLoss, FocalLoss, TverskyLoss
from get_data import Unet2D_DS
from tqdm import tqdm

def train(config):

    torch.cuda.empty_cache()

    # print(f'Parametros: {config['epochs']=}, {batch_size=}, {lr=}, file name={PATH[9:]}\n')

    # Crear datasets #

    ds = Unet2D_DS(config)

    train_size = int(0.8 * len(ds))
    test_size  = len(ds) - train_size

    train_mris, val_mris = random_split(ds, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(
        train_mris, 
        batch_size=config['batch_size'],
        shuffle=True,
    )

    val_dl = DataLoader(
        val_mris, 
        batch_size=config['batch_size'],
    )

    print(f'Tamano del dataset de entrenamiento: {train_size} slices \n')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    unet = Unet(1, depth=5).to(device, dtype=torch.double)
    #print(torch.cuda.memory_summary(device=device, abbreviated=False))

    #criterion = nn.CrossEntropyLoss() # Cross entropy loss performs softmax by default
    #criterion = nn.BCEWithLogitsLoss() # BCEWithLogitsLoss performs sigmoid by default
    #criterion = nn.BCELoss()
    #criterion = DiceLoss()
    criterion = BCEDiceLoss()
    #criterion = FocalLoss()
    #criterion = TverskyLoss()

    optimizer = Adam(unet.parameters(), lr=config['lr'])

    #loss = Variable(loss, requires_grad = True)

    best_loss   = 1.0

    for epoch in tqdm(range(config['epochs'])):  # loop over the dataset multiple times

        #torch.cuda.empty_cache()

        running_loss = 0.0
        running_dice = 0.0
        epoch_loss   = 0.0
        
        print(f'\n\nEpoch {epoch + 1}\n')

        unet.train()
        
        for i, data in enumerate(train_dl, 0):

            inputs, labels = data
            inputs = inputs.unsqueeze(1).to(device, dtype=torch.double)
            labels = labels.to(device, dtype=torch.double)

            outputs = unet(inputs)
            #plot_batch(masks_pred, labels)

            loss = criterion(outputs.double().squeeze(1), labels) # Utilizar esta linea para BCELoss o DiceLoss
            #loss = criterion(outputs, labels.long()) # Utilizar esta linea para Cross entropy loss (multiclase)

            if (i+1) % 48 == 0: 
                print(f'Batch No. {(i+1)} loss = {loss.item():.3f}')

            running_loss += loss.item()
            optimizer.zero_grad() # zero the parameter gradients
            loss.backward(retain_graph=True)#retain_graph=True
            #nn.utils.clip_grad_norm_(unet.parameters(), max_norm=2.0, norm_type=2)
            #nn.utils.clip_grad_value_(unet.parameters(), clip_value=1.0) # Gradient clipping
            optimizer.step()
            
        epoch_loss = running_loss/(i + 1)  

        epoch_dice = 0      

        unet.eval()
        with torch.no_grad():
            print(f'\nValidacion')
            for j, testdata in enumerate(val_dl):
                x, y = testdata
                x = x.unsqueeze(1).to(device, dtype=torch.double)
                y = y.to(device, dtype=torch.double)

                outs  = unet(x)
                #probs = nn.Softmax(dim=1) # Softmax para multiclase
                probs = nn.Sigmoid()  # Sigmoid para biclase
                pval  = probs(outs) 
                preds = torch.where(pval>0.5, 1., 0.)
                #preds = torch.argmax(pval, dim=1)

                batch_dice = dice_coeff(preds.squeeze(1), y)
                epoch_dice += batch_dice.item()
                if (j+1) % 48 == 0: 
                    print(f'\npval min = {pval.min():.3f}, pval max = {pval.max():.3f}')
                    print(f'Batch No. {(i+1)} Dice = {batch_dice.item():.3f}\n')

        epoch_dice = epoch_dice / (j+1) 

        if epoch == 0:
            best_loss = epoch_loss
            best_dice = epoch_dice

        if epoch_loss < best_loss:
            best_loss = epoch_loss

        print(f'\nEpoch loss = {epoch_loss:.3f}, Best loss = {best_loss:.3f}\n')

        if epoch_dice > best_dice:
            best_dice = epoch_dice
            print(f'\nUpdated weights file!')
            torch.save(unet.state_dict(), config['model_fn'])

        print(f'\nEpoch dice (Validation) = {epoch_dice:.3f}, Best dice = {best_dice:.3f}\n')

#     losses.append(head_losses)
#     dices.append(head_dice)

# losses = torch.tensor(losses)
# torch.save(losses, PATH_LOSS)

# dices = torch.tensor(dices)
# torch.save(dices, PATH_DICE)

# print('Finished Training')