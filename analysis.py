import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def assess(config):

    df_loss = pd.read_csv(config['files']['train_Loss']) 
    df_dice_train = pd.read_csv(config['files']['train_Dice'])
    df_dice_test = pd.read_csv(config['files']['test_Dice'])

    mean_losses = df_loss.groupby("Epoca")["Loss"].mean()
    print(f'Average loss last training epoch = {mean_losses[19]}')

    plt.scatter(range(20), mean_losses, marker='o')
    plt.show()

    mean_dices = df_dice_train.groupby("Epoca")["Dice"].mean()
    print(f'Average Dice last training epoch = {mean_dices[19]}')

    plt.scatter(range(20), mean_dices, marker='o')
    plt.show()

    test_dice = df_dice_test["Dice"].mean()
    print(f'Average Dice last training epoch = {test_dice}')
