# Locating-Artifacts-of-Deepfake-Images-in-Frequency-Domain-with-Butterworth-Filter
"the code of paper Locating Artifacts of Deepfake Images in Frequency Domain with Butterworth Filter
>Generative Adversarial Networks (GANs) have
achieved impressive results for many face-swap applications
which can generate realistic images. These synthesized images
can even confuse human eyes. Deepfake images recognition has
been widely researched in the image domain and recognition
based on frequency domain also achieved surprising results.
However, these methods based on frequency directly use the
entire frequency domain. In this paper we transform the deepfake
image into the frequency domain, and use Butterworth filter to
find out the most effective frequency range for identifying the
image as real or fake. We first use high-pass and low-pass filters
to determine the upper and lower boundaries of the effective
frequency band, and then use band-pass filters to narrow the
range. The results show that the effective frequency band is 1-
35, and in the one-dimensional power spectrum of this frequency
band, the real and fake image curves show obvious differences.
In addition, we design a SVM classifier based on the frequency
domain using the effective frequency band we found, and test it
on the mainstream deepfake dataset. The results suggest that all
the mainstream datasets have frequency domain defects, and the
frequency band we find can effectively improve the classification
accuracy."


## datasets
We utilize these three popular datasets:  
FaceForensics++  
UADFV  
DeepFake-TIMIT  
After downloading the data, you need to use extract_test.py to extract the video frame, and then use recogize.py to extract the face

## Experiments
After you have converted the data files you can train a classifer to locate the effective frequency band:  
```
 usage: normalize_deepfake_fillter.py [-h] [-true TRUEDIR] [-false FALSEDIR]
                                     [-t {high,low,bandpass}] [-n NUM_IMAGES]
                                     [-d CUT_OFF [CUT_OFF ...]]
                                     [-w BANDWIDTH [BANDWIDTH ...]]
                                     [-o FILTER_ORDER [FILTER_ORDER ...]]
                                     [--feature_num FEATURE_NUM]
                                     output_path
  positional arguments:
  output_path                 the location of training model

  optional arguments:
  -h, --help            show this help message and exit
  --truedir, -true      Real dataset to load
  --falsedir, -false    Fake dataset to load
  --type, -t            the type of filter to use; Default: high.
  --num_images, -n      the number of images to use; Default: 3000.
  --cut_off, -d         Cut-off frequency of filter; Default: [15,20].
  --bandwidth, -w       Bandwidth of band-pass filter; Default: [10,20].
  --filter_order, -o    the order of filter
  --feature_num         the number of 1D features to use
 
Example: 
python normalize_deepfake_filter.py ./mode/ -t high -n 1000 -d 10 20 -w 10 20 -o 1 5 10 -true ./data/fake_videos/real_images/cropped/ -false ./data/fake_videos/fake_images/cropped/
```
