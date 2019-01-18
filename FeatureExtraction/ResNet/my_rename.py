#encoding=utf-8
# FM,FO,MM,MO
import os
import shutil
from datetime import datetime

def renameAllDirs(dir_path):
    f = open(os.path.join(dir_path,'rename.txt'),'a')
    f.write('rename time {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))+'\n')
    f.write('rename result:\n')
    labels_name = os.listdir(dir_path)
    labels_name.sort()
    i = 0
    for label_name in labels_name:
        label_path = os.path.join(dir_path,label_name)
        if os.path.isdir(label_path):
            new_name = str(i).zfill(3)
            new_label_path = os.path.join(dir_path,new_name)
            shutil.move(label_path,new_label_path)
            f.write('old name: '+label_name+"\t new name: "+new_name+'\n')
            print('old name: '+label_name+"\t new name: "+new_name)
            i+=1

if __name__ == '__main__':
	renameAllDirs('../datas/test_data/')



