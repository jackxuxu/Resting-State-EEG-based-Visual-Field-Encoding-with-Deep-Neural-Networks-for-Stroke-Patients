data=allfu_afterICA2F_rep{1}


cfg = [];
cfg.method     = 'mtmfft'
cfg.taper      = 'hanning'
cfg.foilim     = [1 30];
cfg.keeptrials = 'yes'
freq_segmented = ft_freqanalysis(cfg, data)

begsample = data.sampleinfo(:,1);
endsample = data.sampleinfo(:,2);
time = ((begsample+endsample)/2) / data.fsample;

freq_continuous           = freq_segmented;
freq_continuous.powspctrm = permute(freq_segmented.powspctrm, [2, 3, 1]);
freq_continuous.dimord    = 'chan_freq_time'; % it used to be 'rpt_chan_freq'
freq_continuous.time      = time;             % add the description of the time dimension

cfg = []
cfg.trials=1
cfg.layout = 'GSN-HydroCel-128.sfp'
ft_multiplotER(cfg, data);

cfg = [];
%cfg.baseline = [-0.5 -0.1];
%cfg.baselinetype = 'absolute';
%cfg.zlim = [-1.5e-27 1.5e-27];
cfg.channel = 'E24'; % top figure
figure; ft_singleplotTFR(cfg, freq_continuous);

chanindx = find(strcmp(freq_continuous.label, 'E24'));
figure; imagesc(squeeze(freq_continuous.powspctrm(chanindx,:,:)));



cfg = [];
%cfg.xlim = [0.0 1.0];
%cfg.ylim = [15 20];
%cfg.zlim = [-2e-27 2e-27];
%cfg.baseline = [-0.5 -0.1];
%cfg.baselinetype = 'absolute';
cfg.layout = 'GSN-HydroCel-128.sfp';
figure; ft_topoplotTFR(cfg,freq_segmented); colorbar