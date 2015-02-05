
fprintf(fid, '<table cellspacing="0" cellpadding="1">');

for ii=1:skip:numel(imgInfo)
    fprintf(fid, '<tr>');   
    fprintf(fid, '<td>');
    fprintf(fid, '<font color="red">%s</font>', num2str(ii));
    fprintf(fid, '</td>');
        
    fprintf(fid, '<td>');
        fprintf(fid, '%s', 'red: original, cyan: adjusted to deep pyramid');
        fprintf(fid, '<br>');
        filename = ['http://vision1.cs.ucdavis.edu:1100/weakVideo/cluster_matches2/', ...
            cls, '_img', num2str(ii), '.jpg'];
        fprintf(fid, '<img src="%s" width="100" border="1"/>', filename);    
        fprintf(fid, '<br>');
    fprintf(fid, '</td>'); 
    for jj=1:num_matches
        fprintf(fid, '<td>');
            fprintf(fid, '%f', match_vals(ii,jj));
            fprintf(fid, '<br>');
            filename = ['http://vision1.cs.ucdavis.edu:1100/weakVideo/cluster_matches2/', ...
                cls, '_img', num2str(ii), '_match', num2str(jj), '.jpg'];
            fprintf(fid, '<img src="%s" width="100" border="1"/>', filename);    
            fprintf(fid, '<br>');
        fprintf(fid, '</td>');         
    end
    fprintf(fid, '&nbsp;');
    fprintf(fid, '&nbsp;');
    fprintf(fid, '</tr>');
end
    
fprintf(fid, '</table>');





