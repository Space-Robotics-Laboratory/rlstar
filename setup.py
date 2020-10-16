from setuptools import setup

setup(name='rlstar',
      version='0.0.1',
      install_requires=['gym==0.15.3',
                        'numpy',
                        'scipy',
                        'tqdm',
                        'joblib',
                        'cloudpickle==1.2.2',
                        'click',
                        'opencv-python',
                        "keras==2.3.1"
                        ],
      author="SRL",
      author_email="Tamirblum1@gmail.com"
)

