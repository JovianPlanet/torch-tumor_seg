import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def assess(config):

    print(f"\nAnalisis del modelo: {config['files']['train_Loss']}\n")

    print(f'\nEntrenamiento\n')

    df_loss = pd.read_csv(config['files']['train_Loss']) 
    df_dice_train = pd.read_csv(config['files']['train_Dice'])
    df_dice_val = pd.read_csv(config['files']['val_Dice'])
    df_test = pd.read_csv(config['files']['test_mets'])

    mean_losses = df_loss.groupby("Epoca")["Loss"].mean()
    print(f'Average loss last training epoch = {mean_losses[19]:.3f}\n')

    plt.plot(range(1, 21), mean_losses, marker='o')
    plt.title('Función de costo: BCE + Dice\nÉpoca Vs. Costo')
    plt.xticks(np.arange(1, 21, step=1))
    plt.xlabel(f'Época')
    plt.ylabel(f'BCE+Dice')
    #plt.show()
    plt.savefig(os.path.join(config['plots'], 'loss_plot.pdf'), dpi=300, format='pdf')

    plt.close('all')

    # Dice
    mean_dices = df_dice_train.groupby("Epoca")["Dice"].mean()
    print(f'Average Dice (Entrenamiento) = {mean_dices[19]:.3f}')

    plt.plot(range(1, 21), mean_dices, marker='o', label='Coeficiente Sorensen-Dice (Entrenamiento)')
    plt.title('Coeficiente Sorensen-Dice (Entrenamiento)')
    plt.xticks(np.arange(1, 21, step=1))
    plt.xlabel(f'Época')
    plt.ylabel(f'Dice')
    #plt.show()

    test_dice = df_dice_val["Dice"].mean()
    print(f'Average Dice (test) = {test_dice:.3f}')

    #plt.legend(title='Métricas:')
    plt.savefig(os.path.join(config['plots'], f'train_dice.pdf'), dpi=300, format='pdf')

    plt.close('all')

    #fixCSV(config['files']['val_mets'])

    print(f'\nValidacion\n')

    mean_mets = df_dice_val.groupby("Epoca")["Dice"].mean()
    print(f'Average Dice (Validación) = {mean_mets[19]:.3f}')

    plt.plot(range(1, 21), mean_mets, marker='o', label='Coeficiente Sorensen-Dice (Entrenamiento)')

    plt.title(f'Coeficiente Sorensen-Dice (Validación)')
    plt.xticks(np.arange(1, 21, step=1))
    plt.yticks(np.arange(1.1, step=0.1))
    plt.xlabel(f'Época')
    plt.ylabel(f'Dice')
    #plt.legend(title='Métricas:')
    plt.savefig(os.path.join(config['plots'], f'val_dice.pdf'), dpi=300, format='pdf')

    plt.close('all')

    print(f'\nPrueba\n')

    print(df_test.columns.values)

    mets = list(df_test.columns.values)[1:-1]

    df_melted = pd.melt(df_test, id_vars=['Batch'], value_vars=mets)

    for col in mets:

        if 'Dice' in col:
            mean_mets = df_test[col].mean()
            print(f'Average Dice (Prueba) = {mean_mets:.3f}')
        else:
            continue

    # sns.boxplot(data=df_test['Accuracy'])#, x="Batch", y="value", hue="variable")
    # plt.show()
    
    



