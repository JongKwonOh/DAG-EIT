import yaml
from argparse import ArgumentParser
from scripts import * 

if __name__ == "__main__":
    parser = ArgumentParser(description='Generate PDE file')
    parser.add_argument('--config', type=str, help='Path to config file')
    options = parser.parse_args()
    config_path = options.config
    
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    name = config['data']['name']
        
    if name=='EIT':
        generate_eit(config)

    else:
        print('PDE not found')
        exit(1)