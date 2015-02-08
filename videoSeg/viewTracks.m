clear;


fid = fopen(['/home/SSD1/yjlee-data/projects/weakVideo/YouTube-Objects/car/data/0001/shots/OchsBroxMalik/Results/OchsBroxMalik8_all_0000060.00/001/Tracks201.dat']);
A = textscan(fid, '%f');
A = A{1};
fclose(fid);

numFrames = A(1);
numTracks = A(2);

for ii=1:numFrames
    trackInfo(ii).tracks = [];
end

nn = 3;
for ii=1:numTracks
    trackLabel = A(nn);
    trackLength = A(nn+1);
    nn = nn + 2;
    
    for jj=1:trackLength
        x = A(nn);
        y = A(nn+1);
        frame = A(nn+2);
        nn = nn + 3;
        
        trackInfo(frame+1).tracks = [trackInfo(frame+1).tracks; x y trackLabel];
    end
end

colors = 'rgbycmkwrgbycmkwrgbycmkwrgbycmkwrgbycmkwrgbycmkw';
for ii=1:numFrames
    I = imread(['/home/SSD1/yjlee-data/projects/weakVideo/YouTube-Objects/car/data/0001//shots/001/frame' sprintf('%04d',ii) '.jpg']);
    
    uniqueTrackLabels = unique(trackInfo(ii).tracks(:,3));
    
    figure(1); clf;
    imshow(I);
    hold on;
    for jj=1:numel(uniqueTrackLabels)
        ndx = find(trackInfo(ii).tracks(:,3) == uniqueTrackLabels(jj));
        plot(trackInfo(ii).tracks(ndx,1),trackInfo(ii).tracks(ndx,2), ...
            [colors(uniqueTrackLabels(jj)+1) '*']);
    end
    hold off;
    
%     for jj=1:size(trackInfo(ii).tracks,1)
%         x = trackInfo(ii).tracks(jj,1);
%         y = trackInfo(ii).tracks(jj,2);
%         trackLabel = trackInfo(ii).tracks(jj,3);
%         
%         plot(x,y,[colors(trackLabel+1) '*']);
%     end    
    
    pause(0.1);
end



