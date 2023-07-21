def conf_txt(config):

    text = f"""Dimensiones del modelo  = {config['hyperparams']['model_dims']}
Taza de aprendizaje     = {config['hyperparams']['lr']}        
Numero de epocas        = {config['hyperparams']['epochs']}    
Batch size              = {config['hyperparams']['batch_size']}
Criterion               = {config['hyperparams']['crit']}      
Zooms                   = {config['hyperparams']['new_z']}     
n train                 = {config['hyperparams']['n_train']}   
n val                   = {config['hyperparams']['n_val']}     
n test                  = {config['hyperparams']['n_test']}    
Batch normalization     = {config['hyperparams']['batchnorm']} 
Numero de clases        = {config['hyperparams']['nclasses']}  
Umbral                  = {config['hyperparams']['thres']}"""

    with open(config['files']['params'], 'w') as f:
        f.write(text)


def summary_txt(config, summ):

    with open(config['files']['summary'], 'w') as f:
        f.write(summ)