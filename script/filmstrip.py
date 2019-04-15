import os
from PIL import Image, ImageDraw, ImageFont

def genImg(action, dir, start, length, stride ):
    l = sorted(os.listdir(dir))

    imgs_name = l[start: start+(length*stride): stride]

    img = [Image.open(os.path.join(dir, img_name)) for img_name in imgs_name]
    
    
    width, height = img[0].size
    result = Image.new(img[0].mode, (width * len(img), height+100))
    
    for i, im in enumerate(img):
        result.paste(im, box = (i * width, 0))
    draw = ImageDraw.Draw(result)
    # font = ImageFont.truetype(<font-file>, <font-size>)
    font = ImageFont.truetype("/usr/share/fonts/truetype/tlwg/Garuda.ttf", 40)
    # draw.text((x, y),"Sample Text",(r,g,b))
    draw.text((width * len(img)/2, height),action,(255,255,255), font= font)
    result.save("./examples/{}.png".format(action))

if __name__ == '__main__':
    anno_path = '/home/lq/Documents/Thesis/Thesis/data/TH14_Temporal_annotations_validation/annotation'
    img_path = '/media/lq/C13E-1ED0/dataset/THUMOS/valImg'
    files = os.listdir(anno_path)
    files.remove('Ambiguous_val.txt')
    for file in files:
        action = file.replace("_val.txt",'')
        
        with open(os.path.join(anno_path, file), 'r') as f :
            line = f.readline().strip().split()
            video = line[0]
            startTime = float(line[1])
            start = int(startTime * 30) #30fps
        dir = os.path.join(img_path, video)

        genImg(action, dir, start, 10, 3) 
       