function out = hasnum(x,verbose)
    if nargin <2
        verbose = true;
    end
    Inds= find(~isnan(x(:)),1);
    if isempty(Inds)
        out = false;
    else
        out = true;
        if verbose
        fprintf(' inst rem : %d',numel(find(~isnan(x(:))))/3);
        avNorm = sqrt(sum(abs(x).^2,2));
        avNorm = max(avNorm(:),[],'omitnan');
        fprintf(' Average Norm1 : %f',avNorm);
        end
    end
end