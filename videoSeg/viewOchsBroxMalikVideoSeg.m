clear;
close all;

% add tracks to boundary as a hack to try to fix densification

datadir = '/home/SSD1/yjlee-data/projects/weakVideo/YouTube-Objects/car/data/';

resize_factor = 1/4; %1/4
sample_rate = 8;

% resize_factor = 1; %1/4
% sample_rate = 8;
% subdir = '/shots/OchsBroxMalik/Results/OchsBroxMalik8_all_0000060.00/';
subdir = ['/shots_' num2str(resize_factor) '/OchsBroxMalik/Results/OchsBroxMalik' num2str(sample_rate) '_all_0000060.00/'];

se1 = strel('square',2);
se2 = strel('square',5);

d = dir(datadir);
d = d(3:end);

for ii=9%numel(d)
    dd = dir([datadir d(ii).name '/shots']);
    dd = dd(3:end);

    for jj=1:numel(dd)
        video_dir = [datadir d(ii).name '/shots/' dd(jj).name '/'];
        ddd = dir([video_dir '*.jpg']);
        num_images = numel(ddd);
%             num_images = 10;

        if num_images<=1
            continue;
        end            

        dense_imgs = dir([datadir d(ii).name subdir dd(jj).name '/DenseSegmentation_bdry/*dense.ppm']);    
        overlay_imgs = dir([datadir d(ii).name subdir dd(jj).name '/DenseSegmentation_bdry/*overlay.ppm']);    
        track_imgs = dir([datadir d(ii).name subdir dd(jj).name '/Tracking/*.ppm']);    
        sparse_imgs = dir([datadir d(ii).name subdir dd(jj).name '/SparseSegmentation/*.ppm']);    
        for kk=1:numel(dense_imgs)            
            dense_img = imread([datadir d(ii).name subdir dd(jj).name '/DenseSegmentation_bdry/' dense_imgs(kk).name]);
            overlay_img = imread([datadir d(ii).name subdir dd(jj).name '/DenseSegmentation_bdry/' overlay_imgs(kk).name]);
            track_img = imread([datadir d(ii).name subdir dd(jj).name '/Tracking/' track_imgs(kk).name]);
            sparse_img = imread([datadir d(ii).name subdir dd(jj).name '/SparseSegmentation/' sparse_imgs(kk).name]);
            
            figure(1); clf; 
            subplot(2,3,1); imshow(overlay_img);
            subplot(2,3,2); imshow(dense_img);
            subplot(2,3,3); imshow(track_img);
            subplot(2,3,4); imshow(sparse_img);           
            
            gray_sparse_img = rgb2gray(sparse_img);
            gray_sparse_img(gray_sparse_img==255) = 0;
%             sparse_img_small = gray_sparse_img;            
%             sparse_img_small = gray_sparse_img(sample_rate/2+1:sample_rate:end,sample_rate/2+1:sample_rate:end,:);
            
            count = 1;
            sparse_img_small = zeros([floor(size(gray_sparse_img)/8) 5]);  
            rows = sample_rate/2+1:sample_rate:sample_rate/2+1+sample_rate*(size(sparse_img_small,1)-1);
            cols = sample_rate/2+1:sample_rate:sample_rate/2+1+sample_rate*(size(sparse_img_small,2)-1);
            for mm=-2:1:2
                sparse_img_small(:,:,count) = gray_sparse_img(rows+mm,cols+mm,:);
                count = count + 1;
            end
            sparse_img_small = max(sparse_img_small,[],3);
            
            unique_labels = unique(sparse_img_small);
            unique_labels(unique_labels==255) = [];
            subplot(2,3,5); imagesc(sparse_img_small);
            
            label_num = 1;
            final_sparse_img_small = zeros(size(sparse_img_small));
            for mm=1:numel(unique_labels)
                temp_img = (sparse_img_small==unique_labels(mm));
                temp_img2 = imdilate(temp_img,se1);
                temp_img2 = imerode(temp_img2,se1);
                
%                 temp_img3 = medfilt2(temp_img2,[2 2]);
                CC = bwconncomp(temp_img2,8); 
                temp_img3 = zeros(size(temp_img2));
                for nn=1:CC.NumObjects
                    if numel(CC.PixelIdxList{nn})<=1
                        continue;
                    end
                    temp_img3(CC.PixelIdxList{nn}) = 1;
                end
                ndx = find(temp_img3);
                if isempty(ndx)
                    continue;
                end
                temp_img4 = imdilate(temp_img3,se2);
                
%                 figure(2); clf;
%                 subplot(1,4,1); imagesc(temp_img); 
%                 subplot(1,4,2); imagesc(temp_img2); 
%                 subplot(1,4,3); imagesc(temp_img3); 
%                 subplot(1,4,4); imagesc(temp_img4); 
                                
                temp_img = temp_img4;                
                
                CC = bwconncomp(temp_img,8);  
                num_pixels = zeros(CC.NumObjects,1);
                for nn=1:CC.NumObjects
                    num_pixels(nn) = numel(CC.PixelIdxList{nn});
                end
                [~,max_ndx] = max(num_pixels);
                intersect_pixels = intersect(CC.PixelIdxList{max_ndx},ndx);
%                 if numel(intersect_pixels)<=5
%                     continue;
%                 end
                final_sparse_img_small(intersect_pixels) = label_num;
                label_num = label_num + 1;

%                 CC = bwconncomp(temp_img,8);  
%                 for nn=1:CC.NumObjects
%                     intersect_pixels = intersect(CC.PixelIdxList{nn},ndx);
% %                     if numel(intersect_pixels)<=10 || numel(CC.PixelIdxList{nn})<=37
% %                         continue;
% %                     end                    
%                     final_sparse_img_small(intersect_pixels) = label_num;
%                     label_num = label_num + 1;
%                 end
            end
%             final_sparse_img = final_sparse_img_small;
            final_sparse_img = upsample(final_sparse_img_small,sample_rate);
            final_sparse_img = upsample(final_sparse_img',sample_rate)';
            
            unique_labels = unique(final_sparse_img);
            unique_labels(unique_labels==0) = [];
            figure(1); subplot(2,3,4); 
            for mm=1:numel(unique_labels)
                [yy,xx] = find(final_sparse_img==unique_labels(mm));
                x1 = min(xx)-2;
                y1 = min(yy)-2;
                x2 = max(xx)+2;
                y2 = max(yy)+2;    

                line([x1 x1 x2 x2 x1]',[y1 y2 y2 y1 y1]','color','yellow','linewidth',4);
            end            
         
            pause(0.05);
        end
    end
end