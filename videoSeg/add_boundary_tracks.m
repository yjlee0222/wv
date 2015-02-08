function add_boundary_tracks(filename,I)

fid = fopen(filename,'r');
A = textscan(fid, '%f');
A = A{1};
fclose(fid);

height = size(I,1);
width = size(I,2);

numFrames = A(1);
numTracks = A(2)+length(5:8:width)*2+length(5:8:height)*2;

fid = fopen([filename(1:end-4) '_bdry.dat'],'w');

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





