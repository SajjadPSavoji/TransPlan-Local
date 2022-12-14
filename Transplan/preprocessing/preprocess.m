GoproInfo = load('/home/savoji/Desktop/TransPlan Project/Dataset/GoProHero8CameraParams.mat');
params = GoproInfo.GoProHero8cameraParams.Intrinsics;
v = VideoReader("./../../Dataset/GX010069.MP4");
w = VideoWriter("./../../Dataset/GX010069_undistorted");
open(w);
while(hasFrame(v))
    frame = readFrame(v);
    frame_undist = undistortFisheyeImage(frame, params);
    writeVideo(w,frame_undist);
end
close(w);