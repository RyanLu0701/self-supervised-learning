## Dataset
- Unzip unlabeled data to `./unlabeled`

    ```sh
	Unzip unlabeled data to `./unlabeled`
	and Unzip test data to /test
    ```

- Folder structure
    ```
    .
    ├── test/
    ├── unlabeled/
    ├── Tool
    	      ├────data_parallel_my_v2.py
              ├────KNN.py
              ├────Lookahead.py
              ├────nt_xent.py
              ├────readfile.py
    ├── model_ft
    ├── model_tring    
    ├── main.py    
    ├── model.py     
    ├── Run_test.py       
    ├── readfile.py
    ├── Readme.md
    ├── requirements.txt
    ```

## Environment
- Python 3.7.13 or later version
    ```sh
    pip install -r requirements.txt

    ```

## files
- main.py
use main.py to strat training and predict final output

    ```sh
    python main.py
    ```

-parameter that can be adjust

    ```sh
  	epochs - int ,for training
	epochs_ft - int , for fine tune
	batch_size - int , for data batch
	lr - float m,learning rate
	optimizer - there are two optimizer can be use,one is Lookahead another is Adam
	k  - int , for Lookahead
	alpha - float , for Lookahead
	scheduler_factor -float,for  scheduler
	scheduler_patience -int ,for  scheduler 
	scheduler_min lr - float, for scheduler 
	patience - int , patience for train
	patience_ft - int , patience for fine tune
	lr_ft  -float , Lookahead for fine tune
	k_ft -int , Lookahead for fine tune
	alpha -float , Lookahead for fine tune
	ft_bool -bool, use fine tune or not
    ```

-How to use 

    ```sh
	python main.py --epochs=32 ....
    ```

-model

Model here use Restnet50 and final output change 1000 to 512 

-Run_test 
function for training、finetune and final predict.

-Tool
There are 5 py files here.First file is data_parallel.py this py can help us to achieve Gpu parallel and balance Gpu.
Next KNN.py is to clustering  data ,Lookahead.py is a optimizer,nt_xent.py is a Loss function , and readfile.py is a function to read image file
	

