alldata={}
alldata2={}

for i=1:24
    %data=allpost_afterICA2F_rep{i}
    data=dataori{i}
    for j=1:length(data.trial)
        
        data2=data.trial{j}
     
        
        alldata{j,1}=data2
        alldata{j,2}=i
        
    end 
    
    alldata2=[alldata2;alldata]
       
    
end 
save allpost_afterv6filp alldata2 