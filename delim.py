import os
import os.path
import shutil

for class_dir in os.listdir('dell'):
    os.mkdir('dell/'+class_dir[:4])
    shutil.move('dell/'+class_dir, 'dell/'+class_dir[:4])