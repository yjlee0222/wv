% make_pages
clear;

addpath ('/home/yjlee/projects/weakVideo/html/');
addpath('/home/yjlee/Downloads/VOCdevkit/VOCcode');

VOCinit;
clsNdx = 7;
cls = VOCopts.classes{clsNdx};
skip = 20;
num_matches = 5;

% save cluster images for web viewing
tic;
save_cluster_matches;
toc;

% the php file name to be generated, the variables should be predefined for various php files.
if ~exist(['/home/yjlee/web/weakVideo/php/'],'dir'); 
    mkdir(['/home/yjlee/web/weakVideo/php/']); 
end
fn = ['/home/yjlee/web/weakVideo/php/' cls '_cluster_matches.html'];        
disp(fn);

fid = fopen(fn, 'w');
% generate php header
fprintf(fid, '<html>\n');
fprintf(fid, '<head>\n');
% define css style for the page
style
% java script for form variable verification
% form_verification
fprintf(fid, '</head>\n');
fprintf(fid, '<body><div class=body>\n');

% generate the actual content for the php file
form_cluster_matches

fprintf(fid, '</div></body>\n');
fprintf(fid, '</html>\n');
fclose(fid);

% change file permission so that the php file can be run.
system(['chmod 777 ' fn]);

