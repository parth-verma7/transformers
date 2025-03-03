from pathlib import Path

def get_config():
    return {
        'batch_size': 8,
        'num_epochs': 10,
        'lr': 10**-4,
        'seq_len': 250,
        'lang_src': 'english',
        'lang_tgt': 'hindi',
        'dataset': "Aarif1430/english-to-hindi",
        'model_folder': '/content/drive/MyDrive/transformers/weights',
        'model_basename': 'tmodel_',
        'preload' : None,
        'tokenizer_file': "/content/drive/MyDrive/transformers/tokenizers/tokenizer_{0}.json",
        'experiment_name': "runs/tmodel",
        'd_model': 512
    }

def get_weights_file_path(config, epoch:str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)