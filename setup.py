from setuptools import setup, find_packages

setup(
    name='DocxChain_PO',
    version='0.1',
    packages=find_packages(include=['docxchain', 'docxchain.*']),
    scripts=['docxchain/docxchain.py'],  # Replace 'main.py' with the name of your main file
    # other metadata...
)