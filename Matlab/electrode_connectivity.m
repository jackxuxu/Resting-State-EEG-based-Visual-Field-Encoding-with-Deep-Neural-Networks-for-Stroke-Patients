
for i=1:24
    alldata={}
    for j=1:30
        for k=0:9          
            data=dataori{i}
            step=fix(length(data.trial)/10)+1
            cfg            = [];
            cfg.method     = 'mtmfft';
            cfg.output     = 'fourier';
            cfg.keeptrials = 'yes';
            cfg.tapsmofrq  = 1;
            trial1=k*step+1
            if (k+1)*step<length(data.trial)
                trial2=(k+1)*step
            else
                trial2=length(data.trial)
            end
            cfg.trials     = [trial1:trial2];
            cfg.foilim    = [j j];
            freqcon        = ft_freqanalysis(cfg, data);   
        %         clear freqcon


            cfg         = [];
            cfg.method  ='coh';
            cfg.complex = 'absimag';%imag 'abs' (default), 'angle', 'complex', 'imag', 'real',
            % '-logabs', support for method 'coh', 'csd', 'plv'
            conecitvity_source_con= ft_connectivityanalysis(cfg, freqcon);

            cjh=conecitvity_source_con.cohspctrm([1:128],[1:128]) 
            index=(j-1)*10+k+1
            alldata{index}=cjh
        end
    end
    
    str=num2str(i);
    save(['allcontrol_afterv6connectivity',str],'alldata');
end
%imagesc(cjh)

%cfg           = [];
%cfg.method    = 'degrees';
%cfg.parameter = 'cohspctrm';
%cfg.threshold = .1;
%network_full = ft_networkanalysis(cfg,conecitvity_source_con);



%cfg           = [];
%cfg.method    = 'density';
%cfg.parameter = 'cohspctrm';
%cfg.threshold = .1;
%network_full2 = ft_networkanalysis(cfg,conecitvity_source_con);


 %str = strengths_und(conecitvity_source_con.cohspctrm)
 
 %save allcontrol_afterv6flip alldata2 
