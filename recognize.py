import os
import cv2
import time
import shutil


def getAllPath(dirpath, *suffix):
    PathArray = []
    for r, ds, fs in os.walk(dirpath):
        for fn in fs:
            if os.path.splitext(fn)[1] in suffix:
                fname = os.path.join(r, fn)
                PathArray.append(fname)
    return PathArray


def readPicSaveFace_1(sourcePath, targetPath, invalidPath, *suffix):
    try:
        ImagePaths = getAllPath(sourcePath, *suffix)

        # 对list中图片逐一进行检查,找出其中的人脸然后写到目标文件夹下
        count = 1
        # haarcascade_frontalface_alt.xml为库训练好的分类器文件，下载opencv，安装目录中可找到
        face_cascade = cv2.CascadeClassifier('./opencv-master/data/haarcascades/haarcascade_frontalface_alt.xml')
        for imagePath in ImagePaths:
            try:
                img = cv2.imread(imagePath)

                if type(img) != str:
                    faces = face_cascade.detectMultiScale(img, 1.1, 5)
                    if len(faces):
                        for (x, y, w, h) in faces:
                            # 设置人脸宽度大于16像素，去除较小的人脸
                            if w >= 16 and h >= 16:
                                # 以时间戳和读取的排序作为文件名称
                                listStr = [str(int(time.time())), str(count)]
                                fileName = ''.join(listStr)
                                # 扩大图片，可根据坐标调整
                                X = int(x)
                                W = min(int(x + w), img.shape[1])
                                Y = int(y)
                                H = min(int(y + h), img.shape[0])

                                f = cv2.resize(img[Y:H, X:W], (W - X, H - Y))
                                if f.shape[0] < 100 or f.shape[1] < 100:
                                    count += 1
                                    continue
                                cv2.imwrite(targetPath + os.sep + '%s.jpg' % fileName, f)
                                count += 1
                                print(imagePath + "have face")
                    # else:
                    #   shutil.move(imagePath, invalidPath)
            except:
                continue
    except IOError:
        print("Error")
    else:
        print('Find ' + str(count - 1) + ' faces to Destination ' + targetPath)


if __name__ == '__main__':
    rootdir = './data/DFD/manipulated_sequences/DeepFakeDetection/c40/images/'

    for subdir, dirs, files in os.walk(rootdir):
        print(subdir)
        print(dirs)
        print(files)
        for dir in dirs:
            invalidPath = './data/fake_videos/unused/' + dir + '/'
            os.makedirs(invalidPath, exist_ok=True)
            sourcePath = subdir + dir + '/'
            targetPath1 = subdir + 'cropped/' + dir + '/'
            os.makedirs(targetPath1, exist_ok=True)
            readPicSaveFace_1(sourcePath, targetPath1, invalidPath, '.jpg', '.JPG', 'png', 'PNG')
        print('#')
        break

'''
        for file in files:
            print(file)
            
            invalidPath = './data/fake_videos/unused/0000/'
            os.makedirs('./data/fake_videos/unused/0000/', exist_ok=True)
            sourcePath = './data/fake_videos/real_images/0000/'
            targetPath1 = './data/fake_videos/real_face/0000/'
            os.makedirs('./data/fake_videos/real_face/0000/', exist_ok=True)
            readPicSaveFace_1(sourcePath, targetPath1, invalidPath, '.jpg', '.JPG', 'png', 'PNG')
            '''
