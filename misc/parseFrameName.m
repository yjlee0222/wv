function [video_name,shot_name,frame_name] = parseFrameName(datadir,frame_path)

% need to clean up how this is done...
relevant_frame_name = frame_path(numel(datadir)+1:end);
video_name = relevant_frame_name(1:4);
shot_name = relevant_frame_name(12:14);
frame_name = relevant_frame_name(16:24);

