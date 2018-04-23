#### Overall workflow:
  
  1. Image preprocessing
  2. Model constructing
  3. Training
  4. Validation/Prediction
  5. Ensembling

- Each folder has its own `README` file.

- The Image preprocessing code is put in the folder `utils`.

- We used two pack of codes for modeling, training and predicting: one for python2.7 written by [pkugoodspeed](https://github.com/PKUGoodSpeed) and the other for python3.6 written by [likedan](https://github.com/likedan). They are put in the folders `pkugoodspeed` and `runFinetuneImagenet` respectively.

- The ensemble logics are put in the folder `ensemble`, implemented using jupyter notebooks.
