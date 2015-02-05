
fprintf(fid, '<table cellspacing="0" cellpadding="1">');

for ii=1:numel(clusters)
    fprintf(fid, '<tr>');   
    fprintf(fid, '<td>');
    fprintf(fid, '<font color="red">%s</font>', num2str(ii));
    fprintf(fid, '</td>');
        
    for jj=1:min(10,size(clusters(ii).imNdx,1))
        fprintf(fid, '<td>');
%             fprintf(fid, '<font color="%s">%s</font>', imf.Comment{2}, imf.Comment{1});
            fprintf(fid, '<br>');
            filename = ['http://vision1.cs.ucdavis.edu:1100/weakVideo/clusters/', ...
                cls, '_cluster', num2str(ii), '_img', num2str(jj) '.jpg'];
            fprintf(fid, '<img src="%s" width="100" border="1"/>', filename);    
            fprintf(fid, '<br>');
        fprintf(fid, '</td>');         
    end
    fprintf(fid, '&nbsp;');
    fprintf(fid, '&nbsp;');
    fprintf(fid, '</tr>');
end
    
fprintf(fid, '</table>');

% fprintf(fid, '<?php } ?>\n');
