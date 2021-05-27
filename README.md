# coconet
RNA contact prediction using **Co**evolution and **Co**nvolutional Neural **Net**work.
Its implemented in Python and requires Python version 3.5 or later versions.

# Dependencies
**coconet** uses [pydca](https://github.com/KIT-MBS/pydca) to perform computations on the coevolutionary layer.
You need to install the most recent version (i.e., version 1.23 ) of [pydca](https://github.com/KIT-MBS/pydca). 
By default the command 
```
pip install pydca
```
installs the required version. 
# Usage 
The package can be manually downloaded or cloned using the command  
```bash
git clone  https://github.com/KIT-MBS/coconet
```
## Computing weighted scores
Once  **coconet** is downloaded change to the directory containing file 
`setup.py`  and execute on the command line

```bash
python -m coconet.main <msa_file> --verbose 
```
where `<msa_file>` denotes FASTA formatted multiple sequence alignment (MSA) file of an 
RNA. Note that the first sequence in the MSA file should be the target/reference sequence. 
The optional argument `--verbose` allows logging 
messages printed on the screen. 

By default **coconet** uses a single 3x3 matrix. However, its possible to specify
the matrix size on the command line using the optional argument `msize` as follows.
```bash
python -m coconet.main <msa_file> --msize 5 --verbose 
```

The allowed values of `msize` are 3, 5, and 7.  

In addition, **coconet**  can use two matrices: one for Watson-Crick nucleotide 
pairs and the other for non-Watson-Crick ones. This can be achieved using the 
optional argument `--wc_and_nwc`. For example. 

```bash
python -m coconet.main <msa_file>  --msize 7 --wc_and_nwc --verbose
```
The above command executes  **coconet** using two 7x7 matrices.

In addition, convolution can be performed on top of plmDCA. To enable this feature, use the `--on_plm` optional argument.
Example:
```
python -m coconet.main <msa_file>  --on_plm --num_threads 2 --max_iterations 5000 --verbose
```
The optional arguments `--num_threads` and `--max_iterations` control the numbers of threads used (if OpenMP is supported) and 
gradient decent iterations, respectively. 

Finally, help messages can be prited out on the screen when the command 
```bash
python -m coconet.main
```
is executed, i.e., by running the `coconet.main` module without any additional input from 
the command line.

## Training coconet

Also, the network can be trained on the dataset using a five-fold cross validation procedure. For example, the command
```bash
python -m coconet.train run  --msize 5 --verbose 
```
trains the network using a 5x5 weight matrix using mean-field DCA as a coevolutionary layer. If plmDCA is desired, the `--on_plm` 
optional argument can be provided, for instance as
```bash
python -m coconet.train run --msize 7 --on_plm --num_threads 4 --verbose
```

To see the available arguments to train the network, run the command
```bash
python -m coconet.train
```
