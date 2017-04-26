a = sign(gpuArray.randn(32,32,10,2,'single'))+0;
[Ms,Bs,res,BinMed] = extractMedPCA(a,true,false,[]);
Rate = sum(abs(Bs),5,'omitnan');
Rate = mean(Rate(:))