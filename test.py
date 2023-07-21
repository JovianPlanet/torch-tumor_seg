import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.functional import dice
import pandas as pd
from get_data import Unet2D_DS
from unet import Unet
from metrics import dice_coeff
from utils.plots import plot_batch_full, plot_overlays


def test(config):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing {device} device\n")

    test_ds = Unet2D_DS(config, 'test')

    test_mris = DataLoader(
        test_ds, 
        batch_size=1
    )

    unet = Unet(num_classes=1, depth=5).to(device, dtype=torch.double)
    unet.load_state_dict(torch.load(config['weights']))

    print(f'Test del modelo {config["weights"]}\n')

    acc = BinaryAccuracy(threshold=0.1).to(device, dtype=torch.double)
    dic = Dice(zero_division=1, threshold=0.1).to(device, dtype=torch.double)
    f1s = BinaryF1Score(threshold=0.1).to(device, dtype=torch.double)
    rec = BinaryRecall(threshold=0.1).to(device, dtype=torch.double)
    pre = BinaryPrecision(threshold=0.1).to(device, dtype=torch.double)
    jac = BinaryJaccardIndex(threshold=0.1).to(device, dtype=torch.double)

    gen_dice = 0
    metrics = []

    # Freeze gradients
    with torch.no_grad():
        unet.eval()
        for i, data in enumerate(test_mris):
            images, labels = data
            images = images.unsqueeze(1).to(device, dtype=torch.double)
            labels = labels.to(device, dtype=torch.double)

            # calculate outputs by running images through the network
            outputs = unet(images)
            probs   = nn.Sigmoid()  # Sigmoid para biclase
            preds   = probs(outputs) 
            preds   = torch.where(preds>0.1, 1., 0.)

            '''Metricas''' 
            batch_dice = dice_coeff(preds, labels)
            gen_dice += batch_dice.item()
            tm_dice = dic.forward(outputs, labels.unsqueeze(1).long()).item()
            metrics.append([i, 
                            acc.forward(outputs, labels.unsqueeze(1)).item(),
                            batch_dice.item(),
                            f1s.forward(outputs, labels.unsqueeze(1)).item(),
                            rec.forward(outputs, labels.unsqueeze(1)).item(),
                            pre.forward(outputs, labels.unsqueeze(1)).item(),
                            jac.forward(outputs, labels.unsqueeze(1)).item()]
            )
            '''Fin metricas'''
            if (i+1)%100 == 0:
                print(f'\nMetricas promedio hasta el batch No. {i+1}:')
                print(f'Accuracy      = {acc.compute():.3f}')
                print(f'Dice (custom) = {gen_dice/(i+1):.3f}')
                print(f'Dice (tm)     = {tm_dice.compute():.3f}')
                print(f'F1 Score      = {f1s.compute():.3f}')
                print(f'Sensibilidad  = {rec.compute():.3f}')
                print(f'Precision     = {pre.compute():.3f}')
                print(f'Jaccard       = {jac.compute():.3f}\n')
                

            #plot_batch_full(images.squeeze(1), labels, preds.squeeze(1))

            if torch.any(labels):
                plot_overlays(images.squeeze(1), labels, preds.squeeze(1))
            
    gen_dice = gen_dice / (i+1)

    print(f'\nMetricas totales:')
    print(f'Accuracy      = {acc.compute():.3f}')
    print(f'Dice (custom) = {gen_dice:.3f}')
    print(f'Dice (tm)     = {tm_dice.compute():.3f}')
    print(f'F1 Score      = {f1s.compute():.3f}')
    print(f'Sensibilidad  = {rec.compute():.3f}')
    print(f'Precision     = {pre.compute():.3f}')
    print(f'Jaccard       = {jac.compute():.3f}\n')

    df_metrics = pd.DataFrame(metrics, columns=['Batch', 'Accuracy', 'Dice', 'F1Score', 'Recall', 'Precision', 'Jaccard'])
    df_metrics = df_metrics.assign(id=df_metrics.index.values)
    df_metrics.to_csv(config['test_fn'])

