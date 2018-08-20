import keras.backend as K
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

def get_num_frames_and_duration(video_path):
    ''' Return the number of frames and duration of the video track of the video given '''
    import cv2
    if cv2.__version__ >= '3.0.0':
        CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
        CAP_PROP_FPS = cv2.CAP_PROP_FPS
    else:
        CAP_PROP_FRAME_COUNT = cv2.cv.CV_CAP_PROP_FRAME_COUNT
        CAP_PROP_FPS = cv2.cv.CV_CAP_PROP_FPS

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception('Could not open the video ' + video_path )
    num_frames = int(cap.get(CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(CAP_PROP_FPS))
    duration = num_frames / fps
    return num_frames, duration  







def send_email():
   # import necessary packages
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    import smtplib
    
    # create message object instance
    msg = MIMEMultipart()
      
    message = "AWS done"
    
    # setup the parameters of the message
    with open('password.txt') as p:
        password = p.read()
        print(password)
    msg['From'] = "justforl2.club@gmail.com"
    msg['To'] = "leiqing010@gmail.com"
    msg['Subject'] = "AWS traing done"
    
    # add in the message body
    msg.attach(MIMEText(message, 'plain'))
    
    #create server
    server = smtplib.SMTP('smtp.gmail.com: 587')
    
    server.starttls()
    
    # Login Credentials for sending the mail
    server.login(msg['From'], password)
    
    
    # send the message via the server.
    server.sendmail(msg['From'], msg['To'], msg.as_string())
    
    server.quit()
    
    print ("successfully sent email to %s:" % (msg['To']))

if __name__ == '__main__':
    send_email()
#     path = "/media/lq/C13E-1ED0/dataset/THUMOS/validation/"
#     #print(get_num_frames(path))
#     videos = os.listdir(path)
#     videosInfo = dict()
#     for video in videos:
#         video_path  = path +video
        
#         videosInfo[video] = get_num_frames_and_duration(video_path)
#     print(sys.getsizeof(videosInfo))
# '''
# path = "/media/lq/C13E-1ED0/dataset/THUMOS/validation/video_validation_0000856.mp4"
# print(get_num_frames_and_duration(path))