clear;
close all;

basedir = '/home/SSD1/yjlee-data/projects/weakVideo/YouTube-Objects-2.0/';
cls = 'car';
% load([basedir 'UsefulFiles/' cls '/CandidateTubes_' cls '.mat']);
d = dir([basedir cls '/*.jpg']);

for ii=1:numel(d) 
    figure(1); clf; 
    img = imread([basedir cls '/' d(ii).name]);
    img = imresize(img,1/16);
    imagesc(img);

%     for jj=1:size(CandidateTubes,2)
%         if isempty(T{jj,kk}) 
%             continue;
%         end
% 
%         x1 = CandidateTubes{ii,jj}(1);
%         y1 = CandidateTubes{ii,jj}(2);
%         x2 = CandidateTubes{ii,jj}(3);
%         y2 = CandidateTubes{ii,jj}(4);    
% 
%         line([x1 x1 x2 x2 x1]',[y1 y2 y2 y1 y1]','color','yellow','linewidth',2);
%     end            

    pause(0.05);
end