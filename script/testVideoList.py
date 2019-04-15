import os
import sys
sys.path.insert(0,'../')
from data import videoPaths as path

test_videos_path = path.TEST_VIDEOS_PATH

temporal_test_video_list_path = os.path.join(os.pardir, 'data','TH14_Temporal_Annotations_Test/xgtf_renamed')
output_file_test_videos = os.path.join(os.pardir, 'data', 'test_video_list.txt')
output_file_val_videos = os.path.join(os.pardir, 'data', 'validation_video_list.txt')

def gen_test_list():
    '''
    generate 'test_video_list.txt' file which contians the list of test videos 
    '''
    temporal_test_video_list = os.listdir(temporal_test_video_list_path)
    temporal_test_video_list.sort()

    f = open(output_file_test_videos, 'w')
    for video in temporal_test_video_list:
        videoName = video.split('.')[0]
        f.write(videoName+'.mp4\n')
    f.close()

def compose_download_link_val():
    download_links = os.path.join(os.pardir, 'data', 'download_links_val.txt')
    f1 = open(output_file_val_videos, 'r')
    f2 = open(download_links, 'w')
    videos = f1.read().splitlines()
    from urllib.parse import unquote
    prefixURL = unquote("http://crcv.ucf.edu/THUMOS14/Validation_set/videos/")
    for video in videos:
        link = prefixURL + video
        f2.write(link+'\n')
    f1.close()
    f2.close()

def compose_download_link_test():
    download_links = os.path.join(os.pardir, 'data', 'download_links.txt')
    f1 = open(output_file_test_videos, 'r')
    f2 = open(download_links, 'w')
    videos = f1.read().splitlines()
    from urllib.parse import unquote
    prefixURL = unquote("http://crcv.ucf.edu/THUMOS14/test_set/TH14_test_set_mp4/")
    for video in videos:
        link = prefixURL + video
        f2.write(link+'\n')
    f1.close()
    f2.close()

if __name__ == '__main__':
    # gen_test_list()
    #compose_download_link()
    compose_download_link_val()