alldata={}
alldata2={}
%1-3 delta 8-10 lowalfa 10-12highalfa 4-7 ceta 13-21beta1 22-30beta2
for i=1:24
    %data1=allpre_afterICA2F_rep{i}
    data=dataori{i}
    %cfg=[]
    %cfg.bpfilter='yes'
    %cfg.bpfreq=[1 30]
    %data=ft_preprocessing(cfg,data1)
     for j=1:length(data.trial)
        
        data2=data.trial{j}
     
        
        alldata{j,1}=data2
        alldata{j,2}=i
    end 
    
    alldata2=[alldata2;alldata]
       
    
end 
save allpre_afterv6 alldata2 