to run this on ur local 
first clone this in ur local [install python 3.9.0 don't install new version it won't work with tensorflow install this particular older version]
then activate python virtual envireonment :- 1:- python -m venv venv   2:- venv\Scripts\activate [If this is not installed install it using pip]
then intall the dependecy:-   3:- pip install tensorflow==2.12.0     4:- pip install pillow

Then I wrote the instructions for how to add the dataset pictures in the folders raw and references refer the readme.md
after that run below command

### python train.py [It will start training the modal , it will fail iniitally for some corrupt files in the reaw and references folder if u want u can remove the corrupt file manually it is very much energy draining though,else u can write a script to remove/ignore corrupt files.Use help of ai]
### python enhance.py [It will save the result in the results folder]


