import glob
import numpy as np
import cv2 as cv
import SimpleITK as sitk
class Prepare_Data():
    def __init__(self):
        self.train_images_dir = 'E:\\PyCharmProjects\\TF_tutorials\\tgs-salt-identification-challenge\\train\\imgs\\images\\*'
        self.train_masks_dir = 'E:\\PyCharmProjects\\TF_tutorials\\tgs-salt-identification-challenge\\train\\msks\\masks\\*'
        self.train_image_files=glob.glob(self.train_images_dir)
        self.train_label_files=glob.glob(self.train_masks_dir)
        print("files read",len(self.train_image_files))

    def Crop_to_Tight_Bounds(self, files):
        # path="E:\\PyCharmProjects\\TF_tutorials\\BRATS2015_Training\\BRATS2015_Training\\HGG\\brats_2013_pat0001_1\\VSD.Brain.XX.O.MR_Flair.54512\\VSD.Brain.XX.O.MR_Flair.54512.mha"
        # img0 = sitk.GetArrayFromImage(sitk.ReadImage(path))  # 155,240,240
        # for i in range(img0.shape[0]):
        #     fname='E:\\PyCharmProjects\\TF_tutorials\\Datasets\\brain_ds\\slice_'+str(i)+'.jpg'
        #     if(np.sum(img0)>10):
        #         img=img0[i]
        #         img=cv.resize(img,(200,250))
        #         cv.imwrite(fname,img)

        self.train_image_files=files
        max_left,max_top=240,240
        min_right,min_bottom=0,0
        print('-----', 'left', 'right','top', 'bottom')
        for file in self.train_image_files:
            img=cv.imread(file,0)
            h,w=img.shape
            #print(img.shape)
            for i in range(h): # for left boundary
                colsum=np.sum(img[i,:])
                if colsum>0 and max_left>i:
                    max_left=i
                    break
            for j in reversed(range(h)): # for right boundary
                colsum=np.sum(img[j,:])
                if colsum>0 and min_right<j:
                    min_right=j
                    break
            for k in range(w): # for top boundary
                rowsum=np.sum(img[:,k])
                if rowsum>0 and max_top>k:
                    max_top=k
                    break
            for l in reversed(range(w)): # for bottom boundary
                rowsum=np.sum(img[:,l])
                if rowsum>0 and min_bottom<l:
                    min_bottom=l
                    break
            print('local',max_left, min_right, max_top, min_bottom)
        print('global',max_left,min_right,max_top,min_bottom)
        return max_left,min_right,max_top,min_bottom




if __name__=="__main__":
    pd=Prepare_Data()
    pd.Crop_to_Tight_Bounds()