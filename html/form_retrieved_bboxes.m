
fprintf(fid, '<table cellspacing="0" cellpadding="1">');

for ii=1:numel(unique_img_ids) 
    fprintf(fid, '<tr>');   
        
    fprintf(fid, '<td>');
        fprintf(fid, '%s%f', 'original ', cluster_count(ii,1));
        fprintf(fid, '<br>');
        filename = ['http://vision1.cs.ucdavis.edu:1100/weakVideo/retrieved_bboxes_num_match=' num2str(num_match) '_mirror/', ...
            cls, '_', ids{unique_img_ids(ii)}, '.jpg'];
        fprintf(fid, '<img src="%s" width="100" border="1"/>', filename);    
        fprintf(fid, '<br>');
    fprintf(fid, '</td>'); 
    
    fprintf(fid, '<td>');
        fprintf(fid, '%s%f', 'weighted ', cluster_count(ii,2));
        fprintf(fid, '<br>');
        filename = ['http://vision1.cs.ucdavis.edu:1100/weakVideo/retrieved_bboxes_num_match=' num2str(num_match) '_mirror/', ...
            cls, '_', ids{unique_img_ids(ii)}, '_weighted.jpg'];
        fprintf(fid, '<img src="%s" width="100" border="1"/>', filename);    
        fprintf(fid, '<br>');
    fprintf(fid, '</td>');         
    
    fprintf(fid, '&nbsp;');
    fprintf(fid, '&nbsp;');
    fprintf(fid, '</tr>');
end
    
fprintf(fid, '</table>');





