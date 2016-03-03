
Requirements
-----------
- Python 3.4
- pip (`brew install pip`)
- virtualenv (`pip install virtualenv`)

Install
-------
1. Create and activate virtualenv: `virtualenv .venv/ && source .venv/bin/activate`
2. Install Python packages: `pip install -r requirements.txt`
3. Get the data


This code assumes a `./data` directory that contains the timit data:

```shell
scp -r user@login.hpc.dtu.dk:/dtu-compute/cosound/data/_timit/timit/timit ./data
```
