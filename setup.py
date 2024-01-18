from setuptools import setup, find_packages

with open("README.md", "r") as f:
    description = f.read()

setup(
    name='regresseasy',
    version='0.7',
    packages=find_packages(),
    description='Regression Model Performances',
    long_description=description,
    long_description_content_type='text/markdown',
    author='sai_koushik',
    author_email='saikoushik.gsk@gmail.com',
    url='https://github.com/koushik2299/RegressEasy',
    install_requires=[
        'scikit-learn', 'pandas', 'numpy'  # Corrected dependency names
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
