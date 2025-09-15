from setuptools import setup, find_packages


with open('README.md') as f:
    long_description = f.read()
    

setup(
    name='DSR',
    version='0.1.0',
    description='DSR Modelling',
    url='https://github.com/wenxy123/DSR',
    author='Xiaoyi Wen and Fei Jiang',
    author_email='xiaoyi.wen@outlook.com',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    license="MIT License",
)