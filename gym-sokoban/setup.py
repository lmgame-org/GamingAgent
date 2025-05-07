from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()
    long_description = long_description.replace('(/gym_sokoban/envs/', '(https://github.com/mpSchrader/gym-sokoban/gym_sokoban/envs/')
    long_description = long_description.replace('(/docs/rooms', '(https://github.com/mpSchrader/gym-sokoban/blob/master/docs/rooms')
    long_description = long_description.replace('(/example', '(https://github.com/mpSchrader/gym-sokoban/example')
    #print(long_description)
    

setup(
      name='gym_sokoban',
      version='0.0.5',
      author="Max-Philipp Schrader",
      description='Sokoban environment for OpenAI Gym',
      long_description=long_description,
      long_description_content_type="text/markdown",
       url="https://github.com/mpSchrader/gym-sokoban",
      install_requires=['gym>=0.2.3', 'numpy>=1.14.1', 'tqdm>=4.32.1', 'imageio>=2.3.0', 'requests>=2.22.0'],
      packages=find_packages(),
      package_data={
      'gym_sokoban': ['envs/*', 'envs/surface/*', 'envs/surface/*/*'],
      },
      include_package_data=True,
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
