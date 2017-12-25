import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw
import numpy as np

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

def hog(img):    
    from skimage.feature import hog
    from skimage import color, exposure

    image = color.rgb2gray(img)
    
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()