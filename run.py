import cv2
import numpy as np
import scipy.fftpack as fftpack

def g_pyramid(img, lvl):
    pyramid = [img]
    for i in range(lvl-1):
        img = cv2.pyrDown(img)
        pyramid.append(img)
    return pyramid

def l_pyramid(img, gp):
    res = []
    for i in range(len(gp)-1):
        gaussian_extended = cv2.pyrUp(gp[i+1])
        height, width, _ = gaussian_extended.shape
        gp[i] = cv2.resize(gp[i], (width, height))
        diff = cv2.subtract(gp[i], gaussian_extended)
        res.append(diff)
    res.append(gp[-1])
    return res;

def video_to_face_frames(filename):
    print("convert video to face frames")
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(filename)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_frames = []
    face_rects = ()

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        roi_frame = img

        if len(video_frames) == 0:
            face_rects = faceCascade.detectMultiScale(gray, 1.01, 5)

        if len(face_rects) > 0:
            for (x, y, w, h) in face_rects:
                roi_frame = img[y:y + h, x:x + w]
            if roi_frame.size != img.size:
                roi_frame = cv2.resize(roi_frame, (500, 500))
                video_frames.append(roi_frame)

    cap.release()
    return video_frames, fps

def calc_abs_fft(frames, fps):
    print("calculating fft")
    fft = np.abs(fftpack.fft(frames, axis=0)[0:len(frames)//2])
    freqs = fftpack.fftfreq(len(frames), d=1/fps)[0:len(frames)//2]
    return fft, freqs

def band_pass(fft, freqs, freq_min, freq_max):
    print("band pass, freq_min: " + str(freq_min) + ", freq_max: " + str(freq_max))
    low = np.abs(freqs - freq_min).argmin()
    high = np.abs(freqs - freq_max).argmin()
    fft[:low] = 0
    fft[high:] = 0
    return fft
    
def calc_heart_rate(fft, freqs):
    print(fft.shape)
    print(fft)
    print("calculating heart rate")
    fft_maxes = []
    for i in range(fft.shape[0]):
        fft_maxes.append(fft[i].max())

    tmp_val = 0
    max_amplitude_index = 0
    for i in range(len(fft_maxes)):
        if fft_maxes[i] > tmp_val:
            tmp_val = fft_maxes[i]
            max_amplitude_index = i

    return 60*freqs[max_amplitude_index]


def convert_frames_to_laplacian_pyramid_sequence(frames):
    print("calculatin laplacian pyramids")
    lps = []
    for i,frame in enumerate(frames):
        gp = g_pyramid(frame, 3)
        lp = l_pyramid(frame, gp)
        lps.append(lp[1])

    return lps

def main():
    f = "videos/mk1.mp4"
    frames, fps = video_to_face_frames(f)
    lps = convert_frames_to_laplacian_pyramid_sequence(frames)    
    fft, freqs = calc_abs_fft(lps, fps)
    fft = band_pass(fft, freqs, 1.1, 3) 
    heart_rate = calc_heart_rate(fft, freqs)
    print("estimated heart rate: " + str(heart_rate))

if __name__ == "__main__":
    main()
