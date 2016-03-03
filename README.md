
Requirements
-----------
- Python 3.4

Install
-------
1. Create and activate virtualenv: `pyvenv ./pyenv && source ./pyenv/bin/activate`
2. Install Python packages: `pip install -r requirements.txt`
3. Obtain the data as described in the following section


This code assumes a `./data` directory that contains the timit data:

```shell
scp -r user@login.hpc.dtu.dk:/dtu-compute/cosound/data/_timit/timit/timit ./data
```
