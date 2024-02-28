from setuptools import setup

setup(
    name='hmpdata',
    version='1.0.0',
    # other package info
    entry_points={
        'console_scripts': [
            'make36m=hmpdata.scripts.preprocess_h36m:main',  # 'preprocess' is the command you'll use, and 'preprocess:main' points to the main function in your script
        ],
    }
)
