clear;

addpath('/home/yjlee/Downloads/moseg');
setenv('LD_LIBRARY_PATH','$LD_LIBRARY_PATH:/usr/local/cuda-5.5/lib64:/lib:/home/yjlee/Downloads/moseg');

datadir = '/home/SSD1/yjlee-data/projects/weakVideo/YouTube-Objects/car/data/';

d = dir(datadir);
d = d(3:end);

this_pwd = pwd;
for ii=1:numel(d)
    if isdir([datadir d(ii).name])
        dd = dir([datadir d(ii).name '/shots']);
        dd = dd(3:end);
        
        for jj=1:numel(dd)
            th=tic;            
            
            if exist([datadir d(ii).name '/shots/OchsBroxMalik/Results/OchsBroxMalik8_all_0000060.00/' dd(jj).name],'dir')
                continue;
            end
            
            video_dir = [datadir d(ii).name '/shots/' dd(jj).name '/'];
            ddd = dir([video_dir '*.jpg']);
            num_images = numel(ddd);
%             num_images = 10;
            
            if num_images<=1
                continue;
            end            
            
            cd(video_dir);             
            system('mogrify -format ppm *.jpg');
                        
%             mkdir([video_dir 'OchsBroxMalik']);
            copyfile([video_dir 'shot.bmf'],[video_dir dd(jj).name '.bmf']);
            cd('../');
            system(['/home/yjlee/Downloads/moseg/MoSeg /home/yjlee/projects/weakVideo/videoSeg/filestructure.cfg ' dd(jj).name ' 0 ' num2str(num_images) ' 8']); 
            
            system(['/home/yjlee/Downloads/moseg/dens100gpu /home/yjlee/projects/weakVideo/videoSeg/filestructureDensify.cfg ' dd(jj).name '/image.ppm ' ...
                         'OchsBroxMalik8_all_0000060.00/' dd(jj).name '/Tracks' num2str(num_images) '.dat -1 ' ...
                         'OchsBroxMalik8_all_0000060.00/' dd(jj).name '/DenseSegmentation']);
            
            delete([dd(jj).name '/*.ppm']);         
%             cd(this_pwd);

            fprintf('\n\ndone with video %d/%d, shot %d/%d\n', ii,numel(d),jj,numel(dd));
            fprintf('time taken to complete one shot: %f\n', toc(th));
        end
    end
end
            