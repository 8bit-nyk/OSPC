clc
clear all

a=load('C:\Users\User\Desktop\a.mat'); 
a=a.a;
%% store the exposure 
M=dlmread('C:\Users\User\Desktop\AUB\spring 2022\dataset\calib_narrowGamma_sweep3\times.txt' );
exposures=M(:,3);

%% read crf
crf_truth=dlmread('C:\Users\User\Desktop\AUB\spring 2022\dataset\calib_narrowGamma_sweep3\pcalib.txt' );


A_all=[];
b_all=[];
M1_all=[];
r_all=[];




image1_list=600
image2_list=650


    
%% read  2 images 

image1=imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\calib_narrowGamma_sweep3\00" + num2str(image1_list(1)) + ".jpg");
image2=imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\calib_narrowGamma_sweep3\00" + num2str(image2_list(1)) + ".jpg");


%% 800-890 900-990
%% normalize images

image1=im2double(image1); %./ max(image1(:));
image2=im2double(image2); %./ max(image2(:));

%% filter 
% image1 = medfilt2(image1,[3 3]);
% image2 = medfilt2(image2, [3 3]);
% image2=image2 + a/2;
% image1=image1 + a/2;

image1_exposure=exposures(image1_list(1) + 1);
image2_exposure=exposures(image2_list(1) + 1);

r= (image1_exposure/image2_exposure);
r_all=[r_all r];
%r= 0.93;
total_size=size(image1,1)*size(image1,2);


%% flat
image1_flat=reshape(double(image1),[1 total_size]);   %note
image2_flat=reshape(double(image2),[1 total_size ]);



corr_pts=total_size;

%num_corr_points
 counter=1; 
for i=1:corr_pts
    
    [ M1 , M2]=deal( image1_flat(i), image2_flat(i) );

    [c0,c1,c2]=deal(0,-0.093920939562920,1.093920939562920);
   
if M1>M2 || M1==0 || M2 ==0 || M1==1 || M2 == 1  
continue 
end
    
    invf1= c0 + M1*c1 + M1^2*c2; %+ M1^3*X(4) + M1^4*X(5) + M1^5*X(6) + M1^6*X(7) + M1^7*X(8) + M1^8*X(9) + M1^9*X(10); 
    invf2= c0 + M2*c1 + M2^2*c2;
    exp_est_list(counter)=(invf1/invf2);
    counter=counter+1;

end

mean(exp_est_list)

figure

input=0:1/255:1;
output=c0 + c1*input + c2*input.^2;
plot(input,output)
hold on 
plot(input,crf_truth/255)


