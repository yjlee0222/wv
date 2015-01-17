function imgs = getFrames(cls)

% cls = 'car';
datadir = ['/home/SSD1/yjlee-data/projects/weakVideo/YouTube-Objects/' cls '/data/'];
d = dir(datadir);
d = d(3:end);

n = 1;
clear imgs;
for ii=1:numel(d)
    if isdir([datadir d(ii).name])
        dd = dir([datadir d(ii).name '/shots']);
        dd = dd(3:end);
        
        for jj=1:numel(dd)
            ddd = dir([datadir d(ii).name '/shots/' dd(jj).name '/*.jpg']);
            
            for kk=1:numel(ddd)
                imgs{n,1} = [datadir d(ii).name '/shots/' dd(jj).name '/' ddd(kk).name];
                n = n + 1;
            end
        end
    end
end
