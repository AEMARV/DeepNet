function Inds = indicesOfCol(SZ,count)
    total = sum(count(:));
    [c,r] = meshgrid(1:SZ(2),1:SZ(1));
    if total>0
    Inds = find(r<=count(c),total);
    else
        Inds = find(r<=count(c));
    end
end
