clear;
addpath(genpath('C:\Users\51607\Documents\MATLAB\HRP_REVIS\'))
maindir = 'C:\Users\51607\Documents\MATLAB\HRP_REVIS\';
subdir  = dir( maindir );

subject=[];
indexsubfile=[];
allhrp={};

temphrp={};
%% put all data in one matrix
for m = 1 : length(subdir)
    
     logic=strncmp('.',subdir( m ).name,1);
    
     if logic==1 
         
         subject=[subject,m];
         
     end 

end 

subdir(subject)=[];



for i = 1 : length(subdir)
   
     dirsub=[maindir,subdir( i ).name,'\'];
     
     subfile=dir(dirsub);
     indexsubfile=[];
     for d = 1 : length(subfile)
         
         logic1=strncmp('.',subfile( d ).name,1);
         logic2=strncmp('..',subfile( d ).name,1);
         logic3=strncmp('1',subfile( d ).name,1);
        logic4=strncmp('5',subfile( d ).name,1);
         if logic1||logic2||logic3||logic4==1
             
             indexsubfile=[indexsubfile,d];
             
         end
     end 
      
     subfile(indexsubfile)=[];
         
         
    
     for j=1:length(subfile)
         
         pathsub=[dirsub,subfile(j).name,'\'];
         strpath=fullfile(pathsub,'*.mat');
         filename=dir(strpath);
         temphrp{j,1}=filename;
         
     end
    
      
     
    
     allhrp=[allhrp,temphrp];
   
end
     

      all={};
     for i =1:24
         allnum=[];
         allt=[];
         for j=1:3
             list=allhrp{j,i}
             
             
             filename1=list(1,:).name;
             filename2=list(2,:).name;
             filename3=list(3,:).name;
             
             Test1=load(filename1);
             Test2=load(filename2);
             Test3=load(filename3);
             
             
             r(:,:,1)=Test1.Test.data.dataV.resp;
             r(:,:,2)=Test2.Test.data.dataV.resp;
             r(:,:,3)=Test3.Test.data.dataV.resp;
             
             t(:,:,1)=Test1.Test.data.dataV.RT;
             t(:,:,2)=Test2.Test.data.dataV.RT;
             t(:,:,3)=Test3.Test.data.dataV.RT;
             
             
             r(isnan(r))=0;
             M=mean(r,3);
             RT=nanmean(t,3);
             
             % get index of trials in 3 kinds of areas----------------------------------
             black=find(M==0);
             grey=find(M>0 &M<32);
             white=find(M==32);
             
             brt=nanmean(RT(black));
             grt=nanmean(RT(grey));
             wrt=nanmean(RT(white));
             
             
             temprt(1,1)=brt;
             temprt(1,2)=grt;
             temprt(1,3)=wrt;
             
             
             b=length(black);
             g=length(grey);
             w=length(white);
             tempnum(1,1)=b;
             tempnum(1,2)=g;
             tempnum(1,3)=w;
             
             allt=[allt;temprt];
             allnum=[allnum;tempnum];
         end
         
          allresp{i}=allnum;   % pre post fu resp
          allrt{i}=allt;  % pre post fu  reactime 
         
     end

  
%% pre resp and reaction time 

   prepostfuresp=[];
for j=1:3
    resp=[];
for i=1:24
    
    data=allresp{1,i}(1,:);
    resp=[resp;data];
     
end  
 prepostfuresp=[ prepostfuresp,resp];
end    
     


   prepostfurt=[];
for j=1:3
    rt=[];
for i=1:24
    
    data=allrt{1,i}(1,:);
    rt=[rt;data];
     
end  
 prepostfurt=[ prepostfurt,rt];
end    
     
bedata=[prepostfuresp,prepostfurt]   % all final data for resp and rt
     
     
for i=1:24
    
    data=allresp{i};%% all here before  need to be check again
    
    bppost=(data(2,1)-data(1,1))/data(1,1);
    gppost=(data(2,2)-data(1,2))/data(1,2);
    wppost=(data(2,3)-data(1,3))/data(1,3);
   
    bpfu=(data(3,1)-data(1,1))/data(1,1);
    gpfu=(data(3,2)-data(1,2))/data(1,2);
    wpfu=(data(3,3)-data(1,3))/data(1,3); 
    
    
    postper=[bppost,gppost,wppost];
    fuper=[bpfu,gpfu,wpfu];
    
    perall=[postper;fuper];
    
    hrpper{i}=perall;
    

end 


blackpostad={};
greypostad={}
whitepostad={}
blackfuad={};
greyfuad={}
whitefuad={}

for i =1:24
    data=hrpper{1,i};
    
    bpost=data(1,1);
    blackpostad{1,i}=bpost;
       
    gpost=data(1,2);
    greypostad{1,i}=gpost;
    
    wpost=data(1,3);
    whitepostad{1,i}=wpost;
    
      bfu=data(2,1);
    blackfuad{1,i}=bfu;
       
    gfu=data(2,2);
    greyfuad{1,i}=gfu;
    
    wfu=data(2,3);
    whitefuad{1,i}=wfu; 
    
    hrpfinal=[blackpostad;
greypostad;
whitepostad;
blackfuad;
greyfuad;
whitefuad]  ;

hrpfinal=hrpfinal';
    
end 
  

csvwrite('hrp.csv',hrpfinal);
