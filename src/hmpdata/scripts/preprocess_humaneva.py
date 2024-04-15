import os
import errno
from urllib.request import urlretrieve


def main():
    output_directory = 'datasets'
    dataset_url = 'https://d2zlb0qss9p2br.cloudfront.net/data_3d_humaneva15.npz'
    output_filename = 'data_3d_humaneva15.npz'

    try:
        # Create output directory if it does not exist
        os.makedirs(output_directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    output_file_path = output_directory + '/' + output_filename
    if os.path.exists(output_file_path + '.npz'):
        print('The dataset already exists at', output_file_path + '.npz')
    else:
        print('Downloading HumanEvaI dataset (it may take a while)...')
        humaneva_path = output_directory + '/' + output_filename
        urlretrieve(dataset_url, humaneva_path)
    