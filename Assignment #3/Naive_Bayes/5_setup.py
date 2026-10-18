For Problem 1: Copy and paste the following code for your setup.py file. Only change the regions marked in bold.
import setuptools
with open('README.md', 'r') as f:
 long_description = f.read()
setuptools.setup(
 name=naive_bayes,
 version='0.0.1',
 author='Whitney Syriaque',
 author_email='ws658@georgetown.edu',
 description='A Naive Bayes text classifier with training and prediction methods.',
 long_description=long_description,
 long_description_content_type='text/markdown',
 packages=setuptools.find_packages(),
 python_requires='>=3.6',
 extras_requres={"dev": ["pytest", "flake8", "autopep8"]},

