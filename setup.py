from setuptools import setup, find_packages


with open('README.md', 'r') as fh:
    long_description = fh.read() 

requirements = [
    'pydca==1.22'
]


setup(
    name="coconet",
    version="0.10",
    authors="Mehari B. Zerihun, Fabrizio Pucci and Alexander Schug",
    author_email="mbzerihun@gmail.com",
    python_requires=">=3.5",
    description="RNA contact prediction using coevolution and convolutional neural network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KIT-MBS/coconet",
    packages=find_packages(
        exclude=["*.tests","*.tests.*","tests.*", "tests",
        ],
    ),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Development Status :: 4 - Beta",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    install_requires= requirements,
    tests_require = requirements,
    entry_points={
        "console_scripts":[
            "coconet=coconet.main.run_coconet"
        ],
    },
    #test_suite="tests",
)