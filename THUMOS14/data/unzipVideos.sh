for f in $(cat validation_video_list.txt) ; do 
  echo $f
  unzip -j TH14_validation_set_mp4.zip "*/$f" -d ./validation
done