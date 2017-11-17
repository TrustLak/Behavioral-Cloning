# helper functions:
import cv2

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def combine_gray_images(left, center, right):
    '''
    input: three colored images (depth=3),
    output: one depth=3 numpy array composed of the three grayscale maps of the input images
    '''
    combined_image = left  #initialize the image

    left_gray = rgb2gray(left)
    center_gray = rgb2gray(center)
    right_gray = rgb2gray(right)

    combined_image[:, :, 0] = left_gray
    combined_image[:, :, 1] = center_gray
    combined_image[:, :, 2] = right_gray

    return combined_image


def get_line_images(line, passed_index):

    base_path = ''
    if passed_index:
        base_path = 'train-data/IMG/'
        split_char = '\\'
    else:
        base_path = 'data/IMG/'
        split_char = '/'
    # center
    center_path = line[0]
    filename = center_path.split(split_char)[-1]
    center_path = base_path + filename
    center_img = cv2.imread(center_path)
    
    # left
    left_path = line[1]
    filename = left_path.split(split_char)[-1]
    left_path = base_path + filename

    left_img = cv2.imread(left_path)

    
    # right
    right_path = line[2]
    filename = right_path.split(split_char)[-1]
    right_path = base_path + filename

    right_img = cv2.imread(right_path)

    return (left_img, center_img, right_img)
