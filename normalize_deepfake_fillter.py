import cv2
import os
import radialProfile
from scipy.interpolate import griddata
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from scipy import signal
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from scipy import ndimage
#from skimage import filters
TYPE = ['high', 'low', 'bandpass']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Usage description',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('output_path', type=str, help='Output directory.')
    parser.add_argument('-true', '--truedir', type=str, default=None,
                        help='the dir of real images',
                        )
    parser.add_argument('-false', '--falsedir', type=str, default=None,
                        help='the dir of fake images',
                        )
    parser.add_argument('-t', '--type', type=str, default='high',
                        help='Which filter type, i.e. high, low, bandpasss '
                             'filter for preprocessing',
                        choices=TYPE
                        )
    parser.add_argument('-n', '--num_images', type=int, default=3000,
                        help='Select a number of images number to '
                             "use"
                           )
    parser.add_argument('-d', '--cut-off', nargs='+', type=int, default=[15, 20],
                        help='Cut-off frequency of filter'
                        )
    parser.add_argument('-w', '--bandwidth',nargs='+', type=int, default=[10, 20],
                        help='Bandwidth of filter'
                        )
    parser.add_argument('-o', '--filter_order', nargs='+', type=int, default=[1, 5, 10],
                        help='the order of filter'
                        )
    parser.add_argument('--feature_num', type=int, default=300,
                        help='the number of 1D features to use'
                        )
    args = parser.parse_args()

    return args


def correl2d(img, window):
    s = signal.correlate2d(img, window, mode='same', boundary='fill')
    return s.astype(np.uint8)


def gauss(i, j, sigma):
    return 1/(2 * math.pi *sigma **2) * math.exp(-(i ** 2 + j ** 2) / (2 * sigma ** 2))


def gauss_window(radius, sigma):
    window = np.zeros((radius * 2 + 1, radius * 2 + 1))
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            window[i + radius][j + radius] = gauss(i, j, sigma)
    return window / np.sum(window)


def butterworthPassFilter(image, d, n, cla):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    s1 = np.log(np.abs(fshift))
    def make_transform_matrix(d):
        transfor_matrix = np.zeros(image.shape)
        center_point = tuple(map(lambda x: (x-1)/2, s1.shape))
        for i in range(transfor_matrix.shape[0]):
            for j in range(transfor_matrix.shape[1]):
                def cal_distance(pa, pb):
                    from math import sqrt
                    dis = sqrt((pa[0] - pb[0])**2 + (pa[1] - pb[1])**2)
                    return dis
                dis = cal_distance(center_point, (i, j))
                transfor_matrix[i, j] = 1 / (1 + (dis / d) ** (2*n))
        return transfor_matrix
    if cla == "low":
        d_matrix = make_transform_matrix(d)
    else:
        d_matrix = 1 - make_transform_matrix(d)
    img2 = fshift * d_matrix
    #new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(img2)))
    return img2


# filter
def butterworth_bandstop_kernel(img,D0,W,n=1):
    assert img.ndim == 2
    r,c = img.shape[1],img.shape[0]
    u = np.arange(r)
    v = np.arange(c)
    u, v = np.meshgrid(u, v) #生成网络点坐标矩阵
    low_pass = np.sqrt( (u-r/2)**2 + (v-c/2)**2 ) #相当于公式里的D(u,v),距频谱图矩阵中中心的距离
    kernel = 1-(1/(1+((low_pass*W)/(low_pass**2-D0**2))**(2*n))) #变换公式
    return kernel


def butterworth_bandpass_filter(img,D0,W,n):
    assert img.ndim == 2
    kernel = butterworth_bandstop_kernel(img,D0,W,n)  #得到滤波器
    gray = np.float64(img)  #将灰度图片转换为opencv官方规定的格式
    gray_fft = np.fft.fft2(gray) #傅里叶变换
    gray_fftshift = np.fft.fftshift(gray_fft) #将频谱图低频部分转到中间位置
    #dst = np.zeros_like(gray_fftshift)
    dst_filtered = kernel * gray_fftshift #频谱图和滤波器相乘得到新的频谱图
    #dst_ifftshift = np.fft.ifftshift(dst_filtered) #将频谱图的中心移到左上方
    #dst_ifft = np.fft.ifft2(dst_ifftshift) #傅里叶逆变换
    #dst = np.abs(np.real(dst_ifft))
    #dst = np.clip(dst,0,255)
    return dst_filtered


def testset(d=100, n=1, w0=10, cla="high", output="./", truedir="./true/", falsedir="./false/", number=3000, feature_num = 300):
    data = {}
    epsilon = 1e-8
    N = feature_num
    y = []
    error = []

    number_iter = number

    psd1D_total = np.zeros([number_iter, N])
    label_total = np.zeros([number_iter])
    psd1D_org_mean = np.zeros(N)
    psd1D_org_std = np.zeros(N)

    cont = 0


    # fake data
    rootdir = falsedir

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:

            filename = os.path.join(subdir, file)

            img = cv2.imread(filename, 0)
            # we crop the center
            h = int(img.shape[0] / 3)
            w = int(img.shape[1] / 3)
            img = img[h:-h, w:-w]

            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)

            if cla == "bandpass":
                fshift = butterworth_bandpass_filter(img, d, w0, n)
            else:
                fshift = butterworthPassFilter(img, d, n, cla)

            fshift += epsilon
            magnitude_spectrum = 20 * np.log(np.abs(fshift))
            psd1D = radialProfile.azimuthalAverage(magnitude_spectrum)

            # Calculate the azimuthally averaged 1D power spectrum
            points = np.linspace(0, N, num=psd1D.size)  # coordinates of a
            xi = np.linspace(0, N, num=N)  # coordinates for interpolation

            interpolated = griddata(points, psd1D, xi, method='cubic')
            interpolated /= interpolated[0]

            psd1D_total[cont, :] = interpolated
            label_total[cont] = 0
            cont += 1

            if cont == number_iter:
                break
        if cont == number_iter:
            break

    for x in range(N):
        psd1D_org_mean[x] = np.mean(psd1D_total[:, x])
        psd1D_org_std[x] = np.std(psd1D_total[:, x])

    ## real data
    psd1D_total2 = np.zeros([number_iter, N])
    label_total2 = np.zeros([number_iter])
    psd1D_org_mean2 = np.zeros(N)
    psd1D_org_std2 = np.zeros(N)

    cont = 0
    rootdir2 = truedir

    for subdir, dirs, files in os.walk(rootdir2):
        for file in files:

            filename = os.path.join(subdir, file)
            parts = filename.split("/")

            img = cv2.imread(filename, 0)
            # we crop the center
            h = int(img.shape[0] / 3)
            w = int(img.shape[1] / 3)
            img = img[h:-h, w:-w]

            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)

            if cla == "bandpass":
                fshift = butterworth_bandpass_filter(img, d, w0, n)
            else:
                fshift = butterworthPassFilter(img, d, n, cla)

            fshift += epsilon

            magnitude_spectrum = 20 * np.log(np.abs(fshift))

            # Calculate the azimuthally averaged 1D power spectrum
            psd1D = radialProfile.azimuthalAverage(magnitude_spectrum)

            points = np.linspace(0, N, num=psd1D.size)  # coordinates of a
            xi = np.linspace(0, N, num=N)  # coordinates for interpolation

            interpolated = griddata(points, psd1D, xi, method='cubic')
            interpolated /= interpolated[0]

            psd1D_total2[cont, :] = interpolated
            label_total2[cont] = 1
            cont += 1

            if cont == number_iter:
                break
        if cont == number_iter:
            break

    for x in range(N):
        psd1D_org_mean2[x] = np.mean(psd1D_total2[:, x])
        psd1D_org_std2[x] = np.std(psd1D_total2[:, x])

    y.append(psd1D_org_mean)
    y.append(psd1D_org_mean2)

    error.append(psd1D_org_std)
    error.append(psd1D_org_std2)

    psd1D_total_final = np.concatenate((psd1D_total, psd1D_total2), axis=0)
    label_total_final = np.concatenate((label_total, label_total2), axis=0)

    data["data"] = psd1D_total_final
    data["label"] = label_total_final

    output = open(output + 'test_deepfake.pkl', 'wb')
    pickle.dump(data, output)
    output.close()

    print("Deepfake DATA Saved")


def compute(d=100, n=1, w0=10, cla="high", test=False, output="./", truedir="./true/", falsedir="./false/", number=3000
            , feature_num=300):
    data = {}
    epsilon = 1e-8
    N = feature_num
    y = []
    error = []
    if not test:
        number_iter = number

        psd1D_total = np.zeros([number_iter, N])
        label_total = np.zeros([number_iter])
        psd1D_org_mean = np.zeros(N)
        psd1D_org_std = np.zeros(N)

        cont = 0

        # fake data
        rootdir = falsedir

        for subdir, dirs, files in os.walk(rootdir):
            for file in files:

                filename = os.path.join(subdir, file)

                img = cv2.imread(filename, 0)
                # windows1 = gauss_window(3, 1.0)
                # img = correl2d(img, windows1)
                # n = 3
                # img = ndimage.maximum_filter(img, (n, n))
                # img_sobel = filters.sobel(img)
                # img = img + img_sobel
                # img = img_sobel
                # img_laplace = filters.laplace(img, ksize=3, mask=None)
                # img = img + img_laplace
                # img = img_laplace
                # we crop the center
                h = int(img.shape[0] / 3)
                w = int(img.shape[1] / 3)
                img = img[h:-h, w:-w]

                # f = np.fft.fft2(img)
                # fshift = np.fft.fftshift(f)
                if cla == "bandpass":
                    fshift = butterworth_bandpass_filter(img, d, w0, n)
                else:
                    fshift = butterworthPassFilter(img, d, n, cla)
                fshift += epsilon
                magnitude_spectrum = 20 * np.log(np.abs(fshift))
                psd1D = radialProfile.azimuthalAverage(magnitude_spectrum)

                # Calculate the azimuthally averaged 1D power spectrum
                points = np.linspace(0, N, num=psd1D.size)  # coordinates of a
                xi = np.linspace(0, N, num=N)  # coordinates for interpolation

                interpolated = griddata(points, psd1D, xi, method='cubic')
                interpolated /= interpolated[0]

                psd1D_total[cont, :] = interpolated
                label_total[cont] = 0
                cont += 1

                if cont == number_iter:
                    break
            if cont == number_iter:
                break

        for x in range(N):
            psd1D_org_mean[x] = np.mean(psd1D_total[:, x])
            psd1D_org_std[x] = np.std(psd1D_total[:, x])

        ## real data
        psd1D_total2 = np.zeros([number_iter, N])
        label_total2 = np.zeros([number_iter])
        psd1D_org_mean2 = np.zeros(N)
        psd1D_org_std2 = np.zeros(N)

        cont = 0
        rootdir2 = truedir

        for subdir, dirs, files in os.walk(rootdir2):
            for file in files:

                filename = os.path.join(subdir, file)
                parts = filename.split("/")

                img = cv2.imread(filename, 0)
                # windows1 = gauss_window(3, 1.0)
                # img = correl2d(img, windows1)
                # n = 3
                # img = ndimage.maximum_filter(img, (n, n))
                # img_sobel = filters.sobel(img)
                # img = img + img_sobel
                # img = img_sobel
                # img_laplace = filters.laplace(img, ksize=3, mask=None)
                # img = img + img_laplace
                # img = img_laplace
                # we crop the center
                h = int(img.shape[0] / 3)
                w = int(img.shape[1] / 3)
                img = img[h:-h, w:-w]

                # f = np.fft.fft2(img)
                # fshift = np.fft.fftshift(f)

                if cla == "bandpass":
                    fshift = butterworth_bandpass_filter(img, d, w0, n)
                else:
                    fshift = butterworthPassFilter(img, d, n, cla)
                fshift += epsilon

                magnitude_spectrum = 20 * np.log(np.abs(fshift))

                # Calculate the azimuthally averaged 1D power spectrum
                psd1D = radialProfile.azimuthalAverage(magnitude_spectrum)

                points = np.linspace(0, N, num=psd1D.size)  # coordinates of a
                xi = np.linspace(0, N, num=N)  # coordinates for interpolation

                interpolated = griddata(points, psd1D, xi, method='cubic')
                interpolated /= interpolated[0]

                psd1D_total2[cont, :] = interpolated
                label_total2[cont] = 1
                cont += 1

                if cont == number_iter:
                    break
            if cont == number_iter:
                break

        for x in range(N):
            psd1D_org_mean2[x] = np.mean(psd1D_total2[:, x])
            psd1D_org_std2[x] = np.std(psd1D_total2[:, x])

        y.append(psd1D_org_mean)
        y.append(psd1D_org_mean2)

        error.append(psd1D_org_std)
        error.append(psd1D_org_std2)

        psd1D_total_final = np.concatenate((psd1D_total, psd1D_total2), axis=0)
        label_total_final = np.concatenate((label_total, label_total2), axis=0)

        data["data"] = psd1D_total_final
        data["label"] = label_total_final

        output = open(output + 'train_3200_2.pkl', 'wb')
        pickle.dump(data, output)
        output.close()

        print("DATA Saved")

        # load feature file
        pkl_file = open('train_3200_2.pkl', 'rb')
        data = pickle.load(pkl_file)
        pkl_file.close()
        X = data["data"]
        y = data["label"]

        num = int(X.shape[0] / 2)
        num_feat = X.shape[1]

        psd1D_org_0 = np.zeros((num, num_feat))
        psd1D_org_1 = np.zeros((num, num_feat))
        psd1D_org_0_mean = np.zeros(num_feat)
        psd1D_org_0_std = np.zeros(num_feat)
        psd1D_org_1_mean = np.zeros(num_feat)
        psd1D_org_1_std = np.zeros(num_feat)

        cont_0 = 0
        cont_1 = 0

        # We separate real and fake using the label
        for x in range(X.shape[0]):
            if y[x] == 0:
                psd1D_org_0[cont_0, :] = X[x, :]
                cont_0 += 1
            elif y[x] == 1:
                psd1D_org_1[cont_1, :] = X[x, :]
                cont_1 += 1

        # We compute statistcis
        for x in range(num_feat):
            psd1D_org_0_mean[x] = np.mean(psd1D_org_0[:, x])
            psd1D_org_0_std[x] = np.std(psd1D_org_0[:, x])
            psd1D_org_1_mean[x] = np.mean(psd1D_org_1[:, x])
            psd1D_org_1_std[x] = np.std(psd1D_org_1[:, x])

        # Plot
        x = np.arange(0, num_feat, 1)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, psd1D_org_0_mean, alpha=0.5, color='red', label='Fake', linewidth=2.0)
        ax.fill_between(x, psd1D_org_0_mean - psd1D_org_0_std, psd1D_org_0_mean + psd1D_org_0_std, color='red',
                        alpha=0.2)
        ax.plot(x, psd1D_org_1_mean, alpha=0.5, color='blue', label='Real', linewidth=2.0)
        ax.fill_between(x, psd1D_org_1_mean - psd1D_org_1_std, psd1D_org_1_mean + psd1D_org_1_std, color='blue',
                        alpha=0.2)
        plt.tick_params(axis='x', labelsize=20)
        plt.tick_params(axis='y', labelsize=20)
        ax.legend(loc='best', prop={'size': 20})
        plt.xlabel("Spatial Frequency", fontsize=20)
        plt.ylabel("Power Spectrum", fontsize=20)
        # plt.show()
        plt.savefig(output + 'deepfake_celeb_DF' + cla + '_' + str(d) + '_' + str(w0) + '_' + str(n) + '.png',
                    bbox_inches='tight')

    num = 3
    SVM_r = 0
    precision_SVM_r = 0
    recall_SVM_r = 0
    f1_score_SVM_r = 0
    if test:
        testset(d, n, w0, cla,  output, truedir, falsedir, number, feature_num)
    for z in range(num):
        try:

            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            if test:
                pkl_file2 = open(output + 'test_deepfake.pkl', 'rb')
                data2 = pickle.load(pkl_file2)
                X=data2["data"]
                y=data2["label"]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
            else:
                pkl_file = open(output + 'train_3200_2.pkl', 'rb')
                data = pickle.load(pkl_file)
                pkl_file.close()
                X = data["data"]
                y = data["label"]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
            from sklearn.svm import SVC

            svclassifier_r = SVC(C=6.37, kernel='rbf', gamma=0.86)
            svclassifier_r.fit(X_train, y_train)
            print("train loop " + str(z) + " finished")
            # print('Accuracy on test set: {:.3f}'.format(svclassifier_r.score(X_test, y_test)))

            y_pred = svclassifier_r.predict(X_test)
            n_correct = sum(y_pred == y_test)
            # confusion_matrix(y_train,y_pred)
            # print("准确率: ", n_correct/len(y_pred))
            precision_SVM_r += precision_score(y_test, y_pred)
            recall_SVM_r += recall_score(y_test, y_pred)
            SVM_r += accuracy_score(y_test, y_pred)
            f1_score_SVM_r += f1_score(y_test, y_pred)
        except:
            num -= 1
            print(num)

    print("Average SVM_r: " + str(SVM_r / num))
    print("Precision SVM_r: " + str(precision_SVM_r / num))
    print("Recall SVM_r: " + str(recall_SVM_r / num))
    print("F1_score SVM_r: " + str(f1_score_SVM_r / num))


def main(args):
    print(args)
    d0 = args.cut_off
    width = args.bandwidth
    order = args.filter_order
    type = args.type
    for d in d0:
        for w in width:
            for n in order:
                print("d=", d, "w=", w, "n=", n, "filter: " + type)
                compute(d, n, w, type, True, args.output_path, args.truedir, args.falsedir, args.num_images,
                        args.feature_num)


if __name__ == "__main__":
    args = parse_args()
    main(args)
