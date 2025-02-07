from setuptools import setup, find_packages

setup(
    name='cmdgen',  # Name of your package
    version='2.0.1',  # Version of your package
    packages=find_packages(),  # Automatically find all packages in the directory
    install_requires=[  # Dependencies
        'google-generativeai',
        'rich',
        'pyperclip',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Minimum Python version
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/FreeMarketamilitia/cmdgen',  # Your package's URL
    author='Free Market Militia',
    author_email='FreeMarket@nostates.com',
    license='MIT',  # or any license you prefer
)
