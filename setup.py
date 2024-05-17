from setuptools import setup, find_packages

setup(
    name='hmpdata',
    version='1.6.0',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    # other package info
    entry_points={
        'console_scripts': [
            'make36m=hmpdata.scripts.preprocess_h36m:main',
            'makehumaneva=hmpdata.scripts.preprocess_humaneva:main'
        ],
    }
)
