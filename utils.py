import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import numpy as np

def draw_rectangle(img, regions, filename, save=True, display=False):
    '''
    그릴 직사각형의 영역'들'을 받아, 그려주는 함수
    :param img: 직사각형을 그릴 원본 이미지
    :param regions: 이미지 내에 그릴 직사각형의 영역들의 리스트, 리스트의 각 원소는 [x, y, w, h]로 구성
    :param filename: 그려진 이미지를 저장할 파일 경로 혹은 이름
    :param save: 결과를 저장할지 여부
    :param display: 결과를 보여줄지 여부
    :return: 없음
    '''
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for x, y, w, h in regions:
        rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    if save:
        plt.savefig(filename)

    if display:
        plt.show()

def seperate_region(img, regions):
    '''
    이미지에서 regions 값대로 crop한 결과를 리턴
    :param img: PIL의 image객체
    :param regions: img를 crop할 기준
    :return: crop된 이미지들의 리스트
    '''
    croped_image = []
    for r in regions:
        x, y, w, h = r
        temp_image = img.copy()
        croped_image.append(temp_image.crop((x, y, x+w, y+h)))
    return croped_image

def image_to_tensor(img):
    np_img = np.array(img)
    w, h, c = np_img.shape
    return np_img.reshape((-1, w, h, c))

def refining_ss_regions(ss_regions):
    '''
    selective search의 결과 중 region을 유의미한 img의 영역 부분만 남김
    selective search 라이브러리와 우리 코드와의 호환을 위함
    조건은 코드의 # -1, -2, -3 참고
    :param ss_regions:
    :return:
    '''
    candidates = set()
    for r in ss_regions:
        if r['rect'] in candidates: # -1
            continue
        if r['size'] < 2000: # -2
            continue
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2: # -3
            continue
        candidates.add(r['rect'])

    return np.array(list(candidates))