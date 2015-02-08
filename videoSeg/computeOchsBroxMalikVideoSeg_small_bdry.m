clear;

addpath('/home/yjlee/Downloads/moseg');
addpath('/home/yjlee/projects/weakVideo/videoSeg');
setenv('LD_LIBRARY_PATH','$LD_LIBRARY_PATH:/usr/local/cuda-5.5/lib64:/lib:/home/yjlee/Downloads/moseg');

datadir = '/home/SSD1/yjlee-data/projects/weakVideo/YouTube-Objects/car/data/';
resize_factor = 1/4;
sample_rate = 8;

d = dir(datadir);
d = d(3:end);

this_pwd = pwd;
for ii=9:numel(d)
    if isdir([datadir d(ii).name])
        dd = dir([datadir d(ii).name '/shots']);
        dd = dd(3:end);
        
        for jj=1:numel(dd)
            th=tic;            
            
%             if exist([datadir d(ii).name '/shots_' num2str(resize_factor) '/OchsBroxMalik/Results/OchsBroxMalik' num2str(sample_rate) '_all_0000060.00/' dd(jj).name],'dir')
%                 continue;
%             end
            
            video_dir = [datadir d(ii).name '/shots/' dd(jj).name '/'];
            ddd = dir([video_dir '*.jpg']);
            num_images = numel(ddd);
            
            if num_images<=1
                continue;
            end            
            
            I = imread([video_dir ddd(1).name]);
            filename = [datadir d(ii).name '/shots_' num2str(resize_factor) '/OchsBroxMalik/Results/OchsBroxMalik' num2str(sample_rate) '_all_0000060.00/' dd(jj).name '/Tracks' num2str(num_images) '.dat'];
            add_boundary_tracks(filename,I)

            cd([datadir d(ii).name '/shots_' num2str(resize_factor) '/' dd(jj).name]);             
            system('mogrify -format ppm *.jpg');
            
            cd('../');
            system(['/home/yjlee/Downloads/moseg/dens100gpu /home/yjlee/projects/weakVideo/videoSeg/filestructureDensify.cfg ' dd(jj).name '/image.ppm ' ...
                         'OchsBroxMalik' num2str(sample_rate) '_all_0000060.00/' dd(jj).name '/Tracks' num2str(num_images) '_bdry.dat -1 ' ...
                         'OchsBroxMalik' num2str(sample_rate) '_all_0000060.00/' dd(jj).name '/DenseSegmentation_bdry']);
                                
            delete([dd(jj).name '/*.ppm']); 
            
            fprintf('\n\ndone with video %d/%d, shot %d/%d\n', ii,numel(d),jj,numel(dd));
            fprintf('time taken to complete one shot: %f\n', toc(th));
        end
    end
end
            