from shutil import copyfile
import os

dataDirectory = r'C:\workspace\imageprocessing\plainColor_env\\'

def getOriginalMaskImgName(index, directory=dataDirectory+'fg'):
    prefix = str(index).zfill(4)
    return os.path.join(directory, r'data', prefix+".mask.tiff")

def getStandardMaskImgName(index, subindex, directory=dataDirectory+'fg'):
    prefix = str(index).zfill(4)
    return os.path.join(directory, r'data', prefix+"."+str(subindex)+".mask.tiff")
    
for i in range(3, 59):
    src = getOriginalMaskImgName(i)
    dst = getStandardMaskImgName(i, 0)
    print (src+"---->"+dst)
    copyfile(src, dst)
    
    src = getOriginalMaskImgName(i, dataDirectory+'bg')
    dst = getStandardMaskImgName(i, 0, dataDirectory+'bg')
    print (src+"---->"+dst)
    copyfile(src, dst)