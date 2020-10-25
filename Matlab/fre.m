alldata={}
alldata2={}
for i=1:1
    %data1=allcontrol_afterICA2F_rep_2s{i}
    data1=dataori{i}
    cfg=[]
    cfg.taper='dpss'
    cfg.method='mtmfft'
    cfg.output='powandcsd'
    cfg.keeptrials='yes'

    cfg.tapsmofrq=1
    %cfg.foi=[1:30]
    cfg.foilim     = [1 3]
    freqcon=ft_freqanalysis(cfg,data1)
    for j=1:length(freqcon.cumtapcnt)
        alldata{j,1}=freqcon.powspctrm(j,:,:)
        
    end
    %alldata2=[alldata2;alldata]
    str=num2str(i);
    %save(['allcontrol_afterv6freflip',str],'alldata');
end

