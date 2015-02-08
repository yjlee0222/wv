clear;

datadir = '/home/SSD1/yjlee-data/projects/weakVideo/YouTube-Objects/car/data/';

resize_factor = 1/4; %1/4
sample_rate = 8;

subdir = ['/shots_' num2str(resize_factor) '/OchsBroxMalik/Results/OchsBroxMalik' num2str(sample_rate) '_all_0000060.00/'];

d = dir(datadir);
d = d(3:end);

for ii=9%numel(d)
    dd = dir([datadir d(ii).name '/shots']);
    dd = dd(3:end);

    for jj=1:numel(dd)
        video_dir = [datadir d(ii).name '/shots/' dd(jj).name '/'];
        ddd = dir([video_dir '*.jpg']);
        num_images = numel(ddd);

        if num_images<=1
            continue;
        end            

        sparse_seg_dir = [datadir d(ii).name subdir dd(jj).name '/SparseSegmentation/'];
        tracks = createTubesFromSparseSeg(sparse_seg_dir,sample_rate);
        tracks = getConsistentTubes(tracks);
        
        overlay_imgs = dir([datadir d(ii).name subdir dd(jj).name '/DenseSegmentation_bdry/*overlay.ppm']);         
        for kk=1:numel(tracks)
            overlay_img = imread([datadir d(ii).name subdir dd(jj).name '/DenseSegmentation_bdry/' overlay_imgs(kk).name]);
            figure(1); clf; 
            imshow(overlay_img);   
            for mm=1:numel(tracks(kk).consistent)
                if tracks(kk).consistent(mm)
                    x1 = tracks(kk).bbox_info(mm,1);
                    y1 = tracks(kk).bbox_info(mm,2);
                    x2 = tracks(kk).bbox_info(mm,3);
                    y2 = tracks(kk).bbox_info(mm,4);
                    color_ndx = tracks(kk).bbox_info(mm,5);

                    line([x1 x1 x2 x2 x1]',[y1 y2 y2 y1 y1]','color',colors(color_ndx),'linewidth',4);
                end
            end  
            pause;
        end
    end
end

