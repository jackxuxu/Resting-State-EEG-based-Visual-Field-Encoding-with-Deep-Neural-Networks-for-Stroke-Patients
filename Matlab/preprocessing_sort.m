dataori=allcontrol_afterICA2F_rep_2s
for i=1:24
    
    label=dataori{i}.label
    strall=[]
    for j=1:128
        strlabel= strsplit(string(label(j)),'E' )
        strall(j)=str2num(strlabel(2))
    end
    
    z=[]
    t=[1:128]
    for m=1:128
        tt=find(strall==t(m))

        z(m)=tt

    end 
    
    z=[z,129,130,131]
    
    
    for n=1:length(dataori{i}.trial)
        
       dataori{i}.trial{n}=dataori{i}.trial{n}(z,:)
    end 
    
  
    
    
end

for i=1:24
     load label.mat 
    
    dataori{i}.label=label
end 
    
save data_CONTROL dataori 


