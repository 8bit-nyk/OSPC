clc
clear all



%% store the exposure 
M=dlmread('C:\Users\User\Desktop\AUB\spring 2022\dataset\calib_narrowGamma_sweep3\times.txt' );
exposures=M(:,3);

%% read crf
crf_truth=dlmread('C:\Users\User\Desktop\AUB\spring 2022\dataset\calib_narrowGamma_sweep3\pcalib.txt' );

image1_list=462;
image2_list=463;

% image1_list=start_:1:end_;
% image2_list=start_ + 1:1:end_ + 1;
% image1_list=[image1_list 390:1:400];
% image2_list=[image2_list 391:1:401];
pair=1
for pair=1:1 %size(image1_list,2)
    
%% read  2 images 

image1=imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\calib_narrowGamma_sweep3\00" + num2str(image1_list(pair)) + ".jpg");
image2=imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\calib_narrowGamma_sweep3\00" + num2str(image2_list(pair)) + ".jpg");
% image1=im2double(image1); %./ max(image1(:));
% image2=im2double(image2);
% image2=imgaussfilt(image2,1);
% image1=imgaussfilt(image1,1);
kx=[-1,0,1;
    -2,0,2;
    -1,0,1] ;
ky=[-1,-2,-1;
    0,0,0;
    1,2,1] ;
%Inorm = conv2(double(I), norm, 'same');
%Inorm=uint8(round(Inorm))
%imshow(Inorm)
I=image2;
Ix = conv2(double(I), kx, 'same');
Iy = conv2(double(I), ky, 'same');


%% Find magnitude and orientation of gradient

mag=uint8(round( sqrt(Ix.^2 + Iy.^2) ));


%% normalize images
%image1=im2double(image1); %./ max(image1(:));
%image2=im2double(image2); %./ max(image2(:));

image1_exposure=exposures(image1_list(pair) + 1);
image2_exposure=exposures(image2_list(pair) + 1);

r= (image1_exposure/image2_exposure);
end


a=image1-image2;
