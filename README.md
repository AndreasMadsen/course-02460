
Requirements
-----------
- Python 3.5

Install
-------
1. Install Python packages: `pip install -r requirements.txt`
2. Obtain the data as described in the following section


This code assumes a `./data` directory that contains the timit data and the
elsdsr data which can be fetched as follows:

```shell
scp -r user@login.hpc.dtu.dk:/dtu-compute/cosound/data/_timit/timit/timit ./data/timit
scp -r user@login.hpc.dtu.dk:/dtu-compute/cosound/data/_elsdsr/elsdsr ./data/elsdsr
```
