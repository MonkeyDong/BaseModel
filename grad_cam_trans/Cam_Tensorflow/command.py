import os

command1 = 'python3 main.py --input 1.jpg --output 1s.png --layer_name pool5'
command2 = 'python3 main.py --input 2.jpg --output 2s.png --layer_name pool5'
command3 = 'python3 main.py --input 3.jpg --output 3s.png --layer_name pool5'
command4 = 'python3 main.py --input 4.jpg --output 4s.png --layer_name pool5'

os.system(command1)
os.system(command2)
os.system(command3)
os.system(command4)