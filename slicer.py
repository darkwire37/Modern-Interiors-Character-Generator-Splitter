#Code by AnmolCachra on github: https://github.com/AnmolChachra/Image-Slicer
import numpy as np
import matplotlib.pyplot as plt
import os
 
class ImageSlicer(object):
    
    def __init__(self, source, size, strides=[None, None], BATCH = False, PADDING=False):
        self.source = source
        self.size = size
        self.strides = strides
        self.BATCH = BATCH
        self.PADDING = PADDING
        
    def __read_images(self):
        Images = []
        image_names = sorted(os.listdir(self.source))
        for im in image_names:
            image = plt.imread(os.path.join(dir_path,im))
            Images.append(image)
        return Images

    def __offset_op(self, input_length, output_length, stride):
        offset = (input_length) - (stride*((input_length - output_length)//stride)+output_length)
        return offset
    
    def __padding_op(self, Image):
        if self.offset_x > 0:
            padding_x = self.strides[0] - self.offset_x
        else:
            padding_x = 0
        if self.offset_y > 0:
            padding_y = self.strides[1] - self.offset_y
        else:
            padding_y = 0
        Padded_Image = np.zeros(shape=(Image.shape[0]+padding_x, Image.shape[1]+padding_y, Image.shape[2]),dtype=Image.dtype)
        Padded_Image[padding_x//2:(padding_x//2)+(Image.shape[0]),padding_y//2:(padding_y//2)+Image.shape[1],:] = Image    
        return Padded_Image

    def __convolution_op(self, Image):
        start_x = 0
        start_y = 0
        n_rows = Image.shape[0]//self.strides[0] + 1
        n_columns = Image.shape[1]//self.strides[1] + 1
        small_images = []
        for i in range(n_rows-1):
            for j in range(n_columns-1):
                new_start_x = start_x+i*self.strides[0]
                new_start_y= start_y+j*self.strides[1]
                small_images.append(Image[new_start_x:new_start_x+self.size[0],new_start_y:new_start_y+self.size[1],:])
        return small_images

    def transform(self):
        
        if not(os.path.exists(self.source)):
            raise Exception("Path does not exist!")
            
        else:
            if self.source and not(self.BATCH):
                Image = plt.imread(self.source)
                Images = [Image]
            else: 
                Images = self.__read_images()

            im_size = Images[0].shape
            num_images = len(Images)
            transformed_images = dict()
            Images = np.array(Images)
            
            if self.PADDING:
                
                padded_images = []

                if self.strides[0]==None and self.strides[1]==None:
                    self.strides[0] = self.size[0]
                    self.strides[1] = self.size[1]
                    self.offset_x = Images.shape[1]%self.size[0]
                    self.offset_y = Images.shape[2]%self.size[1]
                    padded_images = list(map(self.__padding_op, Images))
                                         
                elif self.strides[0]==None and self.strides[1]!=None:   
                    self.strides[0] = self.size[0]
                    self.offset_x = Images.shape[1]%self.size[0]
                    if self.strides[1] <= Images.shape[2]:
                        self.offset_y = self.__offset_op(Images.shape[2], self.size[1], self.strides[1])
                    else:
                        raise Exception("stride_y must be between {0} and {1}".format(1,Images.shape[2]))
                    padded_images = list(map(self.__padding_op, Images))

                elif self.strides[0]!=None and self.strides[1]==None:   
                    self.strides[1] = self.size[1]
                    self.offset_y = Images.shape[2]%self.size[1]
                    if self.strides[0] <=Images.shape[1]:
                        self.offset_x = self.__offset_op(Images.shape[1], self.size[0], self.strides[0])
                    else:
                        raise Exception("stride_x must be between {0} and {1}".format(1,Images.shape[1]))
                    padded_images = list(map(self.__padding_op, Images))
                                         
                else:
                    if self.strides[0] > Images.shape[1]:
                        raise Exception("stride_x must be between {0} and {1}".format(1,Images.shape[1]))
                    
                    elif self.strides[1] > Images.shape[2]:
                        raise Exception("stride_y must be between {0} and {1}".format(1,Images.shape[2]))
                        
                    else:
                        self.offset_x = self.__offset_op(Images.shape[1], self.size[0], self.strides[0])
                        self.offset_y = self.__offset_op(Images.shape[2], self.size[1], self.strides[1])
                        padded_images = list(map(self.__padding_op, Images))

                for i, Image in enumerate(padded_images):
                    transformed_images[str(i)] = self.__convolution_op(Image)

            else:
                if self.strides[0]==None and self.strides[1]==None:
                    self.strides[0] = self.size[0]
                    self.strides[1] = self.size[1]

                elif self.strides[0]==None and self.strides[1]!=None:
                    if self.strides[1] > Images.shape[2]:
                        raise Exception("stride_y must be between {0} and {1}".format(1,Images.shape[2]))                 
                    self.strides[0] = self.size[0]

                elif self.strides[0]!=None and self.strides[1]==None:
                    if self.strides[0] > Images.shape[1]:
                        raise Exception("stride_x must be between {0} and {1}".format(1,Images.shape[1]))              
                    self.strides[1] = self.size[1]
                else:
                    if self.strides[0] > Images.shape[1]:
                        raise Exception("stride_x must be between {0} and {1}".format(1,Images.shape[1]))                    
                    elif self.strides[1] > Images.shape[2]:
                        raise Exception("stride_y must be between {0} and {1}".format(1,Images.shape[2]))
                                         
                for i, Image in enumerate(Images):
                    transformed_images[str(i)] = self.__convolution_op(Image)

            return transformed_images
        
    def save_images(self,transformed, save_dir):
        if not(os.path.exists(save_dir)):
            print(save_dir)
            raise Exception("Path does not exist!")
        else:
            for key, val in transformed.items():
                path = save_dir
                #os.mkdir(path)
                for i, j in enumerate(val):
                    plt.imsave(os.path.join(path, str(i+1)+'.png'), j)



# Code by darkwire37 on github (myself)
filename = input("Filename to split: ")
fn = filename
slicer = ImageSlicer(filename,(64,1811))
transf = slicer.transform()
path = os.getcwd()
fl = filename[:-4]
path = os.path.join(path,fl)
os.mkdir(path)
filename = os.path.join(os.path.join("\\split_",filename), path)
slicer.save_images(transf,filename)
for i in range(21, 31):
    os.remove(os.path.join(path,str(i)+".png"))


slicer1 = ImageSlicer(os.path.join(path,"10.png"),(64,384),(64,384))
transf = slicer1.transform()
newPath = os.path.join(path, "pickUp")
os.mkdir(newPath)
slicer1.save_images(transf,newPath)

slicer2 = ImageSlicer(os.path.join(path,"11.png"),(64,320),(64,320))
transf = slicer2.transform()
newPath = os.path.join(path, "setDown")
os.mkdir(newPath)
slicer2.save_images(transf,newPath)

slicer3 = ImageSlicer(os.path.join(path,"12.png"),(64,448),(64,448))
transf = slicer3.transform()
newPath = os.path.join(path, "reach")
os.mkdir(newPath)
slicer3.save_images(transf,newPath)

slicer4 = ImageSlicer(os.path.join(path,"13.png"),(64,448),(64,448))
transf = slicer4.transform()
newPath = os.path.join(path, "reach2")
os.mkdir(newPath)
slicer4.save_images(transf,newPath)

slicer5 = ImageSlicer(os.path.join(path,"17.png"),(64,128),(64,128))
transf = slicer5.transform()
newPath = os.path.join(path, "gunOut")
os.mkdir(newPath)
slicer5.save_images(transf,newPath)
for i in range(5,15):
    os.remove(os.path.join(newPath,str(i)+".png"))

slicer6 = ImageSlicer(os.path.join(path,"19.png"),(64,96),(64,96))
transf = slicer6.transform()
newPath = os.path.join(path, "reload")
os.mkdir(newPath)
slicer6.save_images(transf,newPath)
for i in range(5,19):
    os.remove(os.path.join(newPath,str(i)+".png"))

slicer7 = ImageSlicer(os.path.join(path,"20.png"),(64,96),(64,96))
transf = slicer7.transform()
newPath = os.path.join(path, "damage")
os.mkdir(newPath)
slicer7.save_images(transf,newPath)
for i in range(5,19):
    os.remove(os.path.join(newPath,str(i)+".png"))


for i in range(1,21):
    os.remove(os.path.join(path,str(i)+".png"))

originalPath = os.getcwd()
slicer10 = ImageSlicer(os.path.join(originalPath,fn),(64,192),(64,192))
transf = slicer10.transform()
newPath = os.path.join(path, "6xAnims")
os.mkdir(newPath)
slicer10.save_images(transf,newPath)
path = newPath
for i in range(2, 15):
    os.remove(os.path.join(path,str(i)+".png"))
for i in range(19, 29):
    os.remove(os.path.join(path,str(i)+".png"))
for i in range(33, 43):
    os.remove(os.path.join(path,str(i)+".png"))
for i in range(44, 57):
    os.remove(os.path.join(path,str(i)+".png"))
for i in range(59, 71):
    os.remove(os.path.join(path,str(i)+".png"))
for i in range(73, 85):
    os.remove(os.path.join(path,str(i)+".png"))
for i in range(87, 99):
    os.remove(os.path.join(path,str(i)+".png"))
for i in range(101, 113):
    os.remove(os.path.join(path,str(i)+".png"))
for i in range(121, 183):
    os.remove(os.path.join(path,str(i)+".png"))
for i in range(187, 197):
    os.remove(os.path.join(path,str(i)+".png"))
for i in range(201, 211):
    os.remove(os.path.join(path,str(i)+".png"))
for i in range(215,239):
    os.remove(os.path.join(path,str(i)+".png"))
for i in range(243, 421):
    os.remove(os.path.join(path,str(i)+".png"))


    




