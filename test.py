import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from get_data import Unet2D_DS
from unet import Unet
from metrics import dice_coeff
from utils.plots import *
from torchmetrics.functional import dice


def test(config):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing {device} device\n")

    # dices = []

    PATH_SUPERVISED = 'weights-bcedice-20_eps-100_heads-2023-03-04.pth'

    test_ds = Unet2D_DS(config, 'test')

    test_mris = DataLoader(
        test_ds, 
        batch_size=1#config['batch_size'],
    )

    unet = Unet(num_classes=1, depth=5).to(device, dtype=torch.double)
    unet.load_state_dict(torch.load(PATH_SUPERVISED))

    gen_dice = 0

    # since we're not training, we don't need to calculate the gradients for our outputs
    unet.eval()
    with torch.no_grad():
        for i, data in enumerate(test_mris):
            images, labels = data
            images = images.unsqueeze(1).to(device, dtype=torch.double)
            labels = labels.to(device, dtype=torch.double)

            # calculate outputs by running images through the network
            outputs = unet(images)
            probs = nn.Sigmoid()  # Sigmoid para biclase
            preds  = probs(outputs) 
            preds = torch.where(preds>0.5, 1, 0)

            #batch_dice = dice_coeff(preds, labels)
            batch_dice = dice(preds, labels.long(), ignore_index=0, zero_division=1) # Metrica dice de torchmetrics
            gen_dice += batch_dice
            print(f'Test Dice score (batch): {batch_dice:.3f}')

            #print(f'{images.squeeze(1).shape}, {labels.shape}, {preds.shape}')

            #plot_batch_full(images.squeeze(1), labels, preds.squeeze(1))
            
    gen_dice = gen_dice / (i+1)
    # dices.append(total)
    print(f'\nDice promedio = {gen_dice:.3f}')

# dices = torch.tensor(dices)
# torch.save(dices, PATH_TEST)