import os,shutil
a = os.listdir('../data/train/')
for i in a:
    shutil.move(i,i+'.jpg')
