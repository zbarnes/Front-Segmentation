# Glacier-Front-Classification

This is the code repository for summer research done through the NASA Space Grant Summer 2019 Undergraduate Research Program (SURP).


Right now the file path is hard coded into the seg_coast.py files and will need to be edited with the correct file path. The repository contains a "pickled" version of the segmented image which can be accessed by: 
```python
import pickle

with open('seg_pic', 'rb') as f:
     img = pickle.load(f)
```
Run `make` before running the border removal algorithm in order to compile the required cython code to C.
