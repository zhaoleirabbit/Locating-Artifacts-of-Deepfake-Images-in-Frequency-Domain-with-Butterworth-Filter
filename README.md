# Locating-Artifacts-of-Deepfake-Images-in-Frequency-Domain-with-Butterworth-Filter
"the code of paper Locating Artifacts of Deepfake Images in Frequency Domain with Butterworth Filter
Generative Adversarial Networks (GANs) have
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
