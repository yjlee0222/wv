clear;
close all;

datadir = '/home/SSD1/yjlee-data/projects/weakVideo/YouTube-Objects/car/data/';

resize_factor = 1/4; %1/4
sample_rate = 8;
colors = ['rgbycmwrgbycmwrgbycmwrgbycmwrgbycmwrgbycmkw' ...
    'rgbycmwrgbycmwrgbycmwrgbycmwrgbycmwrgbycmkw' ...
    'rgbycmwrgbycmwrgbycmwrgbycmwrgbycmwrgbycmkw' ...
    'rgbycmwrgbycmwrgbycmwrgbycmwrgbycmwrgbycmkw' ...
    'rgbycmwrgbycmwrgbycmwrgbycmwrgbycmwrgbycmkw' ...
    'rgbycmwrgbycmwrgbycmwrgbycmwrgbycmwrgbycmkw'];
colors = colors(randperm(numel(colors)));

subdir = ['/shots_' num2str(resize_factor) '/OchsBroxMalik/Results/OchsBroxMalik' num2str(sample_rate) '_all_0000060.00/'];

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
        
        dense_segs = dir([datadir d(ii).name subdir dd(jj).name '/DenseSegmentation_bdry/*_dense.ppm']);         
        clear tracks;
        for kk=1:num_images
            dense_seg = imread([datadir d(ii).name subdir dd(jj).name '/DenseSegmentation_bdry/' dense_segs(kk).name]);
            tracks(kk).bbox_info = getTubeBbox(dense_seg);
            tracks(kk).bbox_info(:,1:4) = tracks(kk).bbox_info(:,1:4)./resize_factor;
%             tracks(kk).bbox_info(:,5) = 1;
%             figure(1); imagesc(dense_seg);
%             
% %             % may need to change sparse_seg filename based on num_images..
% %             % right now it's 3 digits, but what if there is more than 999 frames?
%             sparse_seg = rgb2gray(imread([datadir d(ii).name subdir dd(jj).name '/SparseSegmentation/Segments' dense_segs(kk).name(7:end-10) '.ppm']));
%             figure(2); imagesc(sparse_seg);
%             pause%(0.1);
            
%             for mm=1:size(tracks(kk).bbox_info,1)
%                 x1 = tracks(kk).bbox_info(mm,1);
%                 y1 = tracks(kk).bbox_info(mm,2);
%                 x2 = tracks(kk).bbox_info(mm,3);
%                 y2 = tracks(kk).bbox_info(mm,4);
%                 
%                 label_mat = sparse_seg(y1:y2,x1:x2);
%                 figure(1); imagesc(label_mat);
%                 label_mat(label_mat==255) = [];
%                 mode(double(label_mat))
%                 pause;
%             end
        end
        tracks = getConsistentTubes(tracks,3,0.25);
        
        for kk=1:num_images
            figure(1); clf; 
            imshow([video_dir ddd(kk).name]);  
            for mm=1:numel(tracks(kk).consistent)
                x1 = tracks(kk).bbox_info(mm,1);
                y1 = tracks(kk).bbox_info(mm,2);
                x2 = tracks(kk).bbox_info(mm,3);
                y2 = tracks(kk).bbox_info(mm,4);
                color_ndx = tracks(kk).bbox_info(mm,5);
                
                if tracks(kk).consistent(mm)
                    line([x1 x1 x2 x2 x1]',[y1 y2 y2 y1 y1]','color',colors(color_ndx),'linewidth',4);
%                     line([x1 x1 x2 x2 x1]',[y1 y2 y2 y1 y1]','color',colors(mm),'linewidth',4);
                else
                    line([x1 x1 x2 x2 x1]',[y1 y2 y2 y1 y1]','color','k','linewidth',4);                    
                end
            end
            pause(0.02);
        end
    end
end
            
            
            