# **shine**

Requirements:
- Python 3.10
- Git

## **Install project**
```
git clone https://github.com/MarcoParola/shine.git
cd shine
```


Create the virtual environment (we will use venv, but you can use conda)
```
python -m venv env
```

Activate the python virtual environment.

Linux
```
. env/bin/activate
```

Windows
```
.\env\Scripts\activate
```

Install dependecies
```
python -m pip install -r requirements.txt
```

## **Build and import dataset**
Your dataset (composed of images and bounding boxes) has to be load in the ``data`` folder. Create data folder with the following command or create it by gui.
```
mkdir data
```
Move your images in the ``data`` folder.


## **How to use this tool**

To view the images of the dataset, you can run the following command
```
python view_dataset.py
```

To run the detection algorithm according to the *formal property* defined in ``src.formal_methods.py``, you can run the following command
```
python detect.py
```

Finally, to compute the evaluation metrics, you can run the following command
```
python detect.py
```
