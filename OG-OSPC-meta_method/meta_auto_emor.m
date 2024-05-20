
%% load
clear all
clc

B=load('C:\Users\User\Desktop\AUB\spring 2022\codes\meta method\data_hinv\B.mat');

g0=load('C:\Users\User\Desktop\AUB\spring 2022\codes\meta method\data_hinv\g0.mat');

hinv1=load('C:\Users\User\Desktop\AUB\spring 2022\codes\meta method\data_hinv\hinv1.mat');
hinv2=load('C:\Users\User\Desktop\AUB\spring 2022\codes\meta method\data_hinv\hinv2.mat');
hinv3=load('C:\Users\User\Desktop\AUB\spring 2022\codes\meta method\data_hinv\hinv3.mat');
hinv4=load('C:\Users\User\Desktop\AUB\spring 2022\codes\meta method\data_hinv\hinv4.mat');


%% set
g0=g0.g0;
B=B.B;
hinv1=hinv1.hinv1;
hinv2=hinv2.hinv2;
hinv3=hinv3.hinv3;
hinv4=hinv4.hinv4;

%% visualize 
% plot(B,g0)
% hold on
% plot(B,hinv1)
% hold on
% plot(B,hinv2)
% hold on
% plot(B,hinv3)
% hold on
% plot(B,hinv4)

order=60
hinv1_fn = polyfit(B,hinv1,order) ;
hinv2_fn = polyfit(B,hinv2,order) ;
hinv3_fn = polyfit(B,hinv3,order) ;
hinv4_fn = polyfit(B,hinv4,order) ;
g0_fn = polyfit(B,g0,order) ;
%polyval(hinv1_fn,B);
%plot( B, polyval(g0_fn,B) )




%% store the exposure 
M=dlmread('C:\Users\User\Desktop\AUB\spring 2022\dataset\calib_narrowGamma_sweep3\times.txt' );
exposures=M(:,3);

%% read crf
crf_truth=dlmread('C:\Users\User\Desktop\AUB\spring 2022\dataset\calib_narrowGamma_sweep3\pcalib.txt' );

figure(1)
plot(crf_truth)
A_all=[];
b_all=[];
r_all=[];
start_=900;
end_=950;

% image1_list=515:1:520;
% image2_list=516:1:521;

image1_list=start_:1:end_;
image2_list=start_ + 1:1:end_ + 1;

image1_list=[image1_list 515:1:520];
image2_list=[image2_list 516:1:521];

for pair=1:size(image1_list,2)
  
%% read  2 images 

image1=imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\calib_narrowGamma_sweep3\00" + num2str(image1_list(pair)) + ".jpg");
image2=imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\calib_narrowGamma_sweep3\00" + num2str(image2_list(pair)) + ".jpg" );
%% 800-890 900-990
%% normalize images
image1=im2double(image1); %./ max(image1(:));
image2=im2double(image2); %./ max(image2(:));

image1_exposure=exposures(image1_list(pair) + 1);
image2_exposure=exposures(image2_list(pair) + 1);

r= (image1_exposure/image2_exposure);
r_all=[r_all r];
%r= 0.93;
total_size=size(image1,1)*size(image1,2);
% figure('Name','image 1')
% imshow(image1)
% figure('Name','image 2')
% imshow(image2)

%% flat
image1_flat=reshape(double(image1),[1 total_size]); %note
image2_flat=reshape(double(image2),[1 total_size ]);

%% fill the array 

corr_pts=total_size;

[Y,E] = discretize(image1_flat ,10);
counter=1;
if r~=1   %%%%%%
for corr_pt=1:50:corr_pts

    [ M1 , M2]=deal( image1_flat(corr_pt), image2_flat(corr_pt) ) ;
    
    %%%% found a mistake 
    %if M1>0.99  || M2<M1 %%M1 == 1 || M2 ==1 || M1 == 0 || M2 == 0
    if  M1 == 1 || M1>M2  || M1==M2
        
    else
        h1_M1=polyval(hinv1_fn,M1);
        h1_M2=polyval(hinv1_fn,M2);
        
        h2_M1=polyval(hinv2_fn,M1);
        h2_M2=polyval(hinv2_fn,M2);
        
        h3_M1=polyval(hinv3_fn,M1);
        h3_M2=polyval(hinv3_fn,M2);
        
        h4_M1=polyval(hinv4_fn,M1);
        h4_M2=polyval(hinv4_fn,M2);
        
        g0_M1=polyval(g0_fn,M1);
        g0_M2=polyval(g0_fn,M2);
        
     
        %% no weight 
        row = [ (h1_M1 - r*h1_M2) (h2_M1 - r*h2_M2) (h3_M1 - r*h3_M2) (h4_M1 - r*h4_M2)  ];    %(M1^3- r*M2^3)  (M1^4- r*M2^4)]; %(M1^5- r*M2^5) (M1^6- r*M2^6) (M1^7- r*M2^7) (M1^8- r*M2^8) (M1^9- r*M2^9)];
        %row=(1/length_bin)*[ (1 - r) (M1 - r*M2)   (M1^2- r*M2^2)  ];
        element= -g0_M1 + r*g0_M2;
        
        A(counter,:)=row;
        b(counter,1)=element;
        %F(corr_pt)= mult;
        counter=counter+1;

    end


end



A_all=[A_all ; A];
b_all=[b_all ; b];

A=[];
b=[];
end

end

%% solve 

n=4;
cvx_begin

    variable x(n)
    %b=ones(corr_pts,1)
    %minimize( norm(A*x - b) );
    minimize( (A_all*x - b_all )'*(A_all*x - b_all) );
       
cvx_end

%% using pesudo inverse 
cs=inv(A_all'*A_all)*A_all'*b_all;

%rob = robustfit(A_all'*A_all,A_all'*b_all)
%% plot 
%plot 
input=0:1/255:1;

h1_M=polyval(hinv1_fn,B);
h2_M=polyval(hinv2_fn,B);
h3_M=polyval(hinv3_fn,B);
h4_M=polyval(hinv4_fn,B);
g0_M=polyval(g0_fn,B);

finv= g0_M + x(1)*h1_M + x(2)*h2_M + x(3)*h3_M + x(4)*h4_M ;
%finv= g0_M + rob(1)*h1_M + rob(2)*h2_M + rob(3)*h3_M + rob(4)*h4_M ;
figure('Name','j')
plot(B,finv)
hold on

plot(input,crf_truth/255)


%% polyfit 

order=4
p = polyfit(input,crf_truth/255,order);
f1 = polyval(p,input);
plot(input,f1)
hold on
plot(input,crf_truth/255)