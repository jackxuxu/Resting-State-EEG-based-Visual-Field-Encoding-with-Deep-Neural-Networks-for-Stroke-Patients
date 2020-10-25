
alldata={}
alldata2={}

for i=1:24
    %data1=allfu_afterICA2F_rep{i}
    data=dataori{i}
    %cfg=[]
    %cfg.bpfilter='yes'
    %cfg.bpfreq=[22 30]
    %data=ft_preprocessing(cfg,data1)
    for j=1:length(data.trial)
        for k=1:131
            e (k)= entropy(data.trial{j}(k,1:500))
        end
        
     
        
        alldata{j,1}=e
        alldata{j,2}=i
        
    end 
    
    alldata2=[alldata2;alldata]
       
    
end 
save allpost_afterv6entropy alldata2 

