# coconet
RNA contact prediction using **Co**evolution and **Co**nvolutional Neural **Net**work 

# Dependencies
Coconet uses [pydca](https://github.com/KIT-MBS/pydca) to perform computations on the coevolutionary layer.

# Usage 
The package can be manually downloaded or cloned using the command  
```bash
$ git clone  https://github.com/KIT-MBS/coconet
```

Once  **coconet** is downloaded change to the directory containing file 
`setup.py`  and execute on the command line

```bash
$ python -m coconet.main <msa_file> --verbose 
```
where `<msa_file>` denotes FASTA formatted multiple sequence alignment (MSA) file of an 
RNA. The reference/target sequence should be included at the first line of the 
input MSA file, and `--verbose` is an optional argument that  allows logging 
messages printed on the screen. 

By default **coconet** uses a single 3x3 matrix. However, its possible to specify
the matrix size on the command line using the optional argument `msize` as follows.
```bash
$ python -m coconet.main <msa_file> --msize 5 --verbose 
```

The allowed values of `msize` are 3, 5, and 7.  

In addition, **coconet**  can use two matrices: one for Watson-Crick nucleotide 
pairs and the other for non-Watson-Crick ones. This can be achieved using the 
optional argument `--wc_and_nwc`. For example. 

```bash
$ python -m coconet.main <msa_file>  --msize 7 --wc_and_nwc --verbose
```
The above command executes  **cococnet** using two 7x7 matrices.

Finally, help messages can be prited out on the screen when the command 
```bash
$ python -m coconet.main
```
is executed, i.e., by running the `coconet.main` module without any additional input from 
the command line.

