import os
import sys 
sys.path.insert(0,'../')
import constantPaths as path

validation_videos_path = path.VALIDATION_VIDEOS_PATH

temporal_video_list_path = os.path.join(os.pardir, 'data','TH14_Temporal_annotations_validation/TH14_Temporal_annotations_validation_ViperXGTF/xgtf')
output_file = os.path.join(os.pardir, 'data', 'validation_video_list.txt')

def gen_validation_list():
    '''
    generate 'validation_video_list.txt' file which contians the list of validation videos
    '''
    temporal_video_list = os.listdir(temporal_video_list_path)
    temporal_video_list.sort()

    f = open(output_file, 'w')
    for video in temporal_video_list:
        videoName = video.split('.')[0]
        f.write(videoName+'.mp4\n')
    f.close()


def removeUnusefulVideos():
    '''
    Remove unuseful videos

    There are 1,000 videos in total provided for 101 action classes as validation data,
    but only 200 videos for temporal action detection task
    '''
    validation_videos = os.listdir(validation_videos_path)
    
    with open(output_file) as f:
        temporal_videos = f.read().splitlines()

    for video in validation_videos:
        if video in temporal_videos:
            continue
        fileTORemove = os.path.join(validation_videos_path,video)
        os.remove(fileTORemove)


if __name__ == '__main__':
    #gen_validation_list()
    #removeUnusefulVideos()
    print(validation_videos_path)