#去做那该死的json文件
'''
把一个文件夹的图片生成json文件，供mspn做测试
'''
 
import json
import os
from PIL import Image
 
class img2coco(object):
    def __init__(self, save_json_path='./val_fashion_ccyle.json', load_path='./img'):
        '''
        :param labelme_json: 所有labelme的json文件路径组成的列表
        :param save_json_path: json保存位置
        '''
        self.save_json_path = save_json_path
        self.load_path = load_path
        self.images = []
        self.categories = []
        self.annotations = []
        # self.data_coco = {}
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0
        self.imgs = []
        self.save_json()
 
 
    def image(self):
        files = os.listdir(self.load_path)
        j = 0
        for i in files:
            image = {}
            j+=1
            self.imgs.append(self.load_path + i)
            img=Image.open(os.path.join(self.load_path, i))
            width, height = img.size
            image['height'] = height
            image['width'] = width
            image['id'] = j
            image['file_name'] = i
            self.images.append(image)
 
 
    def categorie(self):
        categorie = {}
        categorie['supercategory'] = "person"
        categorie['id'] = 1
        categorie['name'] = "person"
        
        
        categorie['keypoints']=[
                "nose",
                "left_eye",
                "right_eye",
                "left_ear",
                "right_ear",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle"
            ]
        
        categorie['skeleton']=[[16,14],[14,12],[17,15],[15,13],[12,13],
                [6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],[2,3],
                [1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
        
        self.categories.append(categorie)
 
    def annotation(self):
        files = os.listdir(self.load_path)
        j = 0
        for i in files:
            j+=1
            self.imgs.append(self.load_path + i)
            img=Image.open(os.path.join(self.load_path, i))
            width, height = img.size
            annotation = {}
            annotation['segmentation'] = []
            annotation['num_keypoints'] = 17
            annotation['iscrowd'] = 0
            annotation['keypoints'] = [ 
                    0,0,2,
                    0,0,2,
                    0,0,2,
                    0,0,2,
                    0,0,2,
                    0,0,2,
                    0,0,2,
                    0,0,2,
                    0,0,2,
                    0,0,2,
                    0,0,2,
                    0,0,2,
                    0,0,2,
                    0,0,2,
                    0,0,2,
                    0,0,2,
                    0,0,2]
            annotation['image_id'] = j
            # annotation['bbox'] = str(self.getbbox(points)) # 使用list保存json文件时报错（不知道为什么）
            # list(map(int,a[1:-1].split(','))) a=annotation['bbox'] 使用该方式转成list
            annotation['bbox'] = [0, 0, width, height]
            annotation['area'] = 0
            # annotation['category_id'] = self.getcatid(label)
            annotation['category_id'] = 1
            annotation['id'] = j
            self.annotations.append(annotation)
 
    def data2coco(self):
        data_coco = {}
        self.image()
        self.categorie()
        self.annotation()
        data_coco['images'] = self.images
        data_coco['categories'] = self.categories
        data_coco['annotations'] = self.annotations
        return data_coco
 
    def save_json(self):
        self.data_coco = self.data2coco()
        # 保存json文件
        json.dump(self.data_coco, open(self.save_json_path, 'w'), indent=4)  # indent=4 更加美观显示
 
 
img2coco('val_fashion_ccyclegan.json','MSPN_HOME/fashion_0607_result/')

 