import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw
import numpy as np
from xml.etree.ElementTree import Element, SubElement, dump
import xml.dom.minidom

def draw_rectangle(img, region, label='', display=False):
    '''
    그릴 직사각형의 영역 하나를 받아, 그려주는 함수
    :param img: 직사각형을 그릴 원본 이미지, PIL의 Image 객체, 깊은 복사를 통해 원본을 수정하지 않음
    :param region: 이미지 내에 그릴 직사각형의 영역 리스트, 리스트의 각 원소는 x, y, w, h로 구성
    :param label: 직사각형 내에 쓰여질 text
    :param display: 결과를 보여줄지 여부
    :return: 직사각형이 그려진 객체
    '''

    temp_img = img.copy()
    draw = ImageDraw.Draw(temp_img)
    x, y, w, h = region
    draw.rectangle((x, y, x+w, y+h), outline='red')
    draw.text((x,y), label, fill='red')

    if display:
        temp_img.show()

    return temp_img

def seperate_region(img, region, display=False):
    '''
    이미지에서 regions 값대로 crop한 결과를 리턴
    :param img: PIL의 image객체
    :param region: img를 crop할 기준
    :param display: crop된 결과를 보여줄지 여부
    :return: crop된 이미지
    '''
    x, y, w, h = region
    temp_image = img.copy()
    temp_image = temp_image.crop((x, y, x+w, y+h))

    if display:
        temp_image.show()

    return temp_image

def refining_ss_regions(ss_regions):
    '''
    selective search의 결과 중 region을 유의미한 img의 영역 부분만 남김
    selective search 라이브러리와 우리 코드와의 호환을 위함
    조건은 코드의 # -1, -2, -3 참고
    :param ss_regions: selective_search의 결과 regions
    :return: regions 중, 유의미한 결과의 numpy array
    '''
    candidates = set()
    for r in ss_regions:
        if r['rect'] in candidates: # -1
            continue
        if r['size'] < 2000 or r['size'] > 10000: # -2
            continue
        x, y, w, h = r['rect']
        if w / h > 2.5 or h / w > 2.5 : # -3
            continue
        if min(w, h) < 20:
            continue
        candidates.add(r['rect'])

    return np.array(list(candidates))

def CNN_classifier(img, softmax_classifier, input_size, label_dic, boundary):
    softmax_score = softmax_classifier(img)[0]
    ind = np.argmax(softmax_score)
    if softmax_score[ind] > boundary:
        return list(label_dic.keys())[ind]
    return None

def hog(img, pixels_per_cell = (16, 16), save = False, save_dir = 'temp.jpg'):
    '''
    img: Image 객체를 np로 변형한 객체
    return
        fd: img의 hog value np객체(1차원)
        hog_image_rescaled: img의 hog value를 이미지화 시킨 np객체
    '''
    from skimage.feature import hog
    from skimage import color, exposure

    image = color.rgb2gray(img)
    
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=pixels_per_cell, cells_per_block=(1, 1), visualise=True)
    #print(fd.size)
    #hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

    return fd

def hog_to_3d(hog, size):
    hog = hog.reshape(size)
    hog = np.transpose(hog, (2,3,1))
    return hog
    '''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # Rescale histogram for better display

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()
    '''

def grayscale(img_set):
    RGB_to_L = np.array([[[[0.299,0.587,0.114]]]])
    img_set = np.sum(img_set * RGB_to_L, axis=3, keepdims=True)
    return img_set

def box_to_xml(filename, path, size, box_set):
    note = Element("annotation")
    SubElement(note, "folder").text = 'Images'
    SubElement(note, "filename").text = filename
    SubElement(note, "path").text = path
    source = Element("source")
    SubElement(source, "database").text = 'Unknown'
    note.append(source)
    size_ele = Element('size')
    SubElement(size_ele, 'width').text = str(size[0])
    SubElement(size_ele, 'height').text = str(size[1])
    SubElement(size_ele, 'depth').text = str(size[2])
    note.append(size_ele)
    SubElement(note, 'segmented').text = '0'
    for box in box_set:
        obj = Element('object')
        SubElement(obj, 'name').text = 'block'
        SubElement(obj, 'pose').text = 'Unspecified'
        SubElement(obj, 'truncated').text = '0'
        SubElement(obj, 'difficult').text = '0'
        bndbox = Element('bndbox')
        SubElement(bndbox, 'xmin').text = str(box[0])
        SubElement(bndbox, 'ymin').text = str(box[1])
        SubElement(bndbox, 'xmax').text = str(box[0]+box[2])
        SubElement(bndbox, 'ymax').text = str(box[1]+box[3])
        obj.append(bndbox)
        note.append(obj)

    indent(note)
    return note

def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
    
    return elem

def list_to_str(ls):
    result = ''
    for line in ls:
        for ob in line:
            result = result + str(ob) + ' '
        result = result[:-1] + '\n'
    return result[:-1]

def box_to_txt(labels, box_set, image_size):
    result = []
    dw, dh = image_size
    for i, box in enumerate(box_set):
        x = (box[0]+box[2]/2)/dw
        y = (box[1]+box[3]/2)/dh
        w = box[2]/dw
        h = box[3]/dh
        result.append([labels[i], x, y, w, h])
    return list_to_str(result)