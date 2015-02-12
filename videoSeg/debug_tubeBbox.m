clear;
close all;

addpath('/home/yjlee/projects/weakVideo/videoSeg');

datadir = '/home/SSD1/yjlee-data/projects/weakVideo/YouTube-Objects/car/data/';

resize_factor1 = 1/4; %1/2
resize_factor2 = 1/2; %1/2
sample_rate = 8;

subdir1 = ['/shots_' num2str(resize_factor1) '/OchsBroxMalik/Results/OchsBroxMalik' num2str(sample_rate) '_all_0000060.00/'];
subdir2 = ['/shots_' num2str(resize_factor2) '/OchsBroxMalik/Results/OchsBroxMalik' num2str(sample_rate) '_all_0000060.00/'];

d = dir(datadir);
d = d(3:end);

for ii=1:numel(d)
    dd = dir([datadir d(ii).name '/shots']);
    dd = dd(3:end);

    for jj=1:numel(dd)
        video_dir = [datadir d(ii).name '/shots/' dd(jj).name '/'];
        ddd = dir([video_dir '*.jpg']);
        num_images = numel(ddd);

        if num_images<=1
            continue;
        end            

        dense_segs1 = dir([datadir d(ii).name subdir1 dd(jj).name '/DenseSegmentation_bdry/*dense.ppm']);    
        dense_segs2 = dir([datadir d(ii).name subdir2 dd(jj).name '/DenseSegmentation_bdry/*dense.ppm']);    
        for kk=1:numel(dense_segs1)            
            I1 = imresize(imread([video_dir ddd(kk).name]),resize_factor1);        
            I2 = imresize(imread([video_dir ddd(kk).name]),resize_factor2);            
        
            dense_seg1 = imread([datadir d(ii).name subdir1 dd(jj).name '/DenseSegmentation_bdry/' dense_segs1(kk).name]);
            dense_seg2 = imread([datadir d(ii).name subdir2 dd(jj).name '/DenseSegmentation_bdry/' dense_segs2(kk).name]);
            
            figure(1); clf; 
            subplot(2,2,1); imshow(dense_seg1);            
            subplot(2,2,3); imshow(I1);            
            tube_bbox = getTubeBbox(dense_seg1);
            for mm=1:size(tube_bbox,1)                
                x1 = tube_bbox(mm,1);
                y1 = tube_bbox(mm,2);
                x2 = tube_bbox(mm,3);
                y2 = tube_bbox(mm,4); 
                line([x1 x1 x2 x2 x1]',[y1 y2 y2 y1 y1]','color','yellow','linewidth',4);
            end   
            
            subplot(2,2,2); imshow(dense_seg2);         
            subplot(2,2,4); imshow(I2);         
            tube_bbox = getTubeBbox(dense_seg2);
            for mm=1:size(tube_bbox,1)                
                x1 = tube_bbox(mm,1);
                y1 = tube_bbox(mm,2);
                x2 = tube_bbox(mm,3);
                y2 = tube_bbox(mm,4); 
                line([x1 x1 x2 x2 x1]',[y1 y2 y2 y1 y1]','color','yellow','linewidth',4);
            end   
            
            pause(0.03);
        end
    end
end