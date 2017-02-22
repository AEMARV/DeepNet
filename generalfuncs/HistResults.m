function  [] = HistResults(ch,layer,res) 
try 
figure(2); 
catch  
    histogramRes = figure; 
end 
histogram(res(layer).x(:,:,ch,1),300,'Normalization','pdf');title('forward') 
try 
figure(3); 
catch 
    histogramBackRes = figure; 
end 
histogram(res(layer-1).dzdx(1,1,ch,:),300,'Normalization','pdf');title('Backward') 
end