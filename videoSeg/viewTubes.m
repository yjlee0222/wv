clear;

datadir = '/home/SSD1/yjlee-data/projects/weakVideo/YouTube-Objects/car/data/';

d = dir(datadir);
d = d(3:end);

for ii=1:numel(d)
    if ~isdir([datadir d(ii).name])
        continue;
    end
    
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
        
        load([video_dir 'Tpercentile3.mat'],'T')       
        for kk=1:num_images 
            figure(1); clf; 
            img = imread([video_dir ddd(kk).name]);
            imagesc(img);
            
            for mm=1:size(T,1)
                if isempty(T{mm,kk}) 
                    continue;
                end
                
                x1 = T{mm,kk}(1);
                y1 = T{mm,kk}(2);
                x2 = T{mm,kk}(3);
                y2 = T{mm,kk}(4);    
                
                line([x1 x1 x2 x2 x1]',[y1 y2 y2 y1 y1]','color','yellow','linewidth',2);
            end            
            
            pause(0.01);
        end
    end
end
