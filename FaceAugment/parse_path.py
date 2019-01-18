#encoding=utf-8
from glob import glob
import os

def get_filelist(input_img_path,tmp_img_path):
    fileList = []
    outputFolder = tmp_img_path
    num_already,num_update=0,0
    # 1.首先获取到所有的输入文件夹里面的图片
    img_paths = glob(os.path.join(input_img_path,"*.jpg"))
    # 2.检查哪些是已经转换过得图片
    already_augment_lable = os.listdir(tmp_img_path)
    # 3.添加到文件列表
    for img_path in img_paths:
        img_name = os.path.split(img_path)[1]
        if img_name[:-4] not in already_augment_lable:
            single_file = str(img_name.split(".")[0])+","+str(img_path)+",None"
            fileList.append(single_file)
            num_update+=1
        else:
            num_already+=1
            print img_name,'is already augment'
    print 'already augment imgs num is',num_already,'Update Imgs is',num_update
    return fileList,num_update