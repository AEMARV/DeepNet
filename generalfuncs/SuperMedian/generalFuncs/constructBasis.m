function w = constructBasis(wn,Norm)
Dim = size(wn,1);
if size(wn,2) == Dim
    w = wn;
    return;
end
w = wn;
proposal = 0;
while norm(proposal,Norm)==0
    proposal = rand(Dim,1);
    projected = w'*proposal;
    proposal = proposal - sum(projected' .* w,2);
    
end
proposal = proposal /norm(proposal,Norm);
w = cat(2,wn,proposal);
if size(w,2) < Dim
    w = constructBasis(w,Norm);
else
    return;
end
end