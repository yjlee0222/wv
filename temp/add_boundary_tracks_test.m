clear;

datadir = '/home/SSD1/yjlee-data/projects/weakVideo/YouTube-Objects/car/data/';

% copyfile([datadir '0001/shots/001/BroxMalikResults/Tracks201.dat'],...
%     [datadir '0001/shots/001/BroxMalikResults/Tracks201_copy.dat']);

fid = fopen([datadir '0001/shots/001/BroxMalikResults/Tracks201.dat'],'r');
A = textscan(fid, '%f');
A = A{1};
fclose(fid);

I = imread([datadir '0001/shots/001/frame' sprintf('%04d',1) '.jpg']);
height = size(I,1);
width = size(I,2);

numFrames = A(1);
numTracks = A(2)+length(5:8:width)*2 + length(5:8:height)*2;

fid = fopen([datadir '0001/shots/001/BroxMalikResults/Tracks201_copy.dat'],'w');

fprintf(fid,'%d %d\n',numFrames,numTracks);
nn = 3;
for ii=1:A(2)
    trackLength = A(nn+1);
    fprintf(fid,'%d %d\n',A(nn),A(nn+1));
    nn = nn + 2;    
    for jj=1:trackLength
        fprintf(fid,'%f %f %d\n',A(nn),A(nn+1),A(nn+2));
        nn = nn + 3;
    end
end

trackLabel = 1000;
trackLength = numFrames;

col_ndx = 5:8:width;
col_ndx = col_ndx([1 end]);
row_ndx = 5:8:height;
row_ndx = row_ndx([1 end]);

for x=5:8:width
    for y=row_ndx  
        fprintf(fid,'%d %d\n',trackLabel,trackLength);

        for frame=1:trackLength        
            fprintf(fid,'%d %d %d\n',x,y,frame-1);        
        end
    end
end
for y=5:8:height
    for x=col_ndx  
        fprintf(fid,'%d %d\n',trackLabel,trackLength);

        for frame=1:trackLength        
            fprintf(fid,'%d %d %d\n',x,y,frame-1);        
        end
    end
end
fclose(fid);

fid = fopen([datadir '0001/shots/001/BroxMalikResults/Tracks201_copy.dat']);
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
    I = imread([datadir '0001/shots/001/frame' sprintf('%04d',ii) '.jpg']);
    
    uniqueTrackLabels = unique(trackInfo(ii).tracks(:,3));
    
    figure(1); clf;
    imshow(I);
    hold on;
    for jj=1:numel(uniqueTrackLabels)
        ndx = find(trackInfo(ii).tracks(:,3) == uniqueTrackLabels(jj));
        try
            plot(trackInfo(ii).tracks(ndx,1),trackInfo(ii).tracks(ndx,2), ...
                [colors(uniqueTrackLabels(jj)+1) '*']);
        catch
            plot(trackInfo(ii).tracks(ndx,1),trackInfo(ii).tracks(ndx,2), ...
                ['w*']);
        end
    end
    hold off;
    
    pause(0.1);
end



