function overlaps = computeOverlap(boxes1, boxes2)

overlaps = zeros(size(boxes1, 1), size(boxes2, 1));
if isempty(boxes1)
  overlaps = [];
else
  x11 = boxes1(:,1);
  y11 = boxes1(:,2);
  x12 = boxes1(:,3);
  y12 = boxes1(:,4);
  areab1 = (x12-x11+1) .* (y12-y11+1);
  x21 = boxes2(:,1);
  y21 = boxes2(:,2);
  x22 = boxes2(:,3);
  y22 = boxes2(:,4);
  areab2 = (x22-x21+1) .* (y22-y21+1);

  for i = 1 : size(boxes1, 1)
    for j = 1 : size(boxes2, 1)
      xx1 = max(x11(i), x21(j));
      yy1 = max(y11(i), y21(j));
      xx2 = min(x12(i), x22(j));
      yy2 = min(y12(i), y22(j));
      w = xx2-xx1+1;
      h = yy2-yy1+1;
      if w > 0 && h > 0
        overlaps(i, j) = w * h / (areab1(i) + areab2(j) - w * h);
      end
    end
  end  
end
