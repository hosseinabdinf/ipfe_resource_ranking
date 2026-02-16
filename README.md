# ASRR-IPFE
A Simple Resource Ranking using Inner-Product Functional Encryption


# Requirement
To use this script, you need to follow the instructions below:


1. Make sure you have **Python** and **pip** installed on your system:

        $ python3 -V
        $ pip -V

- if you don't have them, check this [link](https://packaging.python.org/en/latest/tutorials/installing-packages/).

2. Make the python virtual environment and active it:

        $ python3 -m venv venv
        $ source venv/bin/activate 

3. Install the required packages:

        pip install -r requirements.txt  

## Run Ranking Script

To run the ranking script, do as follows:
        
    $ python3 ranking.py

## References
This script uses the [pymife](https://pypi.org/project/pymife/) package. 
We used the Simple DDH version to make the first version, and we refer 
you to check the original paper for more details. 

1. [Selective Secure DDH based scheme](https://eprint.iacr.org/2015/017.pdf)