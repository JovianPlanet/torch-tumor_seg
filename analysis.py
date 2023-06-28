import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def assess(config):

    print(f"\nAnalisis del modelo: {config['files']['train_Loss']}\n")

    df_loss = pd.read_csv(config['files']['train_Loss']) 
    df_dice_train = pd.read_csv(config['files']['train_Dice'])
    df_dice_test = pd.read_csv(config['files']['test_Dice'])

    mean_losses = df_loss.groupby("Epoca")["Loss"].mean()
    print(f'Average loss last training epoch = {mean_losses[19]:.3f}')

    plt.scatter(range(20), mean_losses, marker='o')
    plt.title('BCE Dice loss')
    plt.show()

    mean_dices = df_dice_train.groupby("Epoca")["Dice"].mean()
    print(f'Average Dice (eval) = {mean_dices[19]:.3f}')

    plt.scatter(range(20), mean_dices, marker='o')
    plt.title('Dice coefficient (eval)')
    plt.show()

    test_dice = df_dice_test["Dice"].mean()
    print(f'Average Dice (test) = {test_dice:.3f}')