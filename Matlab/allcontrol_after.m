alldata={}
alldata2={}
data3=[]
for i=1:24
    data=allcontrol_afterICA2F_rep_2s{i}
    %for j=1:length(data.trial)
    parfor j=1:10
      %  data2=data.trial{j}
        if size(data.trial{j})==[131 500]
            data3=cat(3,data3,data.trial{j})
        end
    end 
end 
save data3 data3
save allcontrol_afterv6 alldata2 