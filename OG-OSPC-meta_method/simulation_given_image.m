clc
clear all

%% read inv crf truth
I_truth=dlmread('C:\Users\User\Desktop\AUB\spring 2022\dataset\calib_narrowGamma_sweep3\pcalib.txt' );

%% polynomial fit to inv crf => irradiance
% input is intensities M 
% output is irradiances I
M = 0:1/255:1;

order = 15;
I_poly = polyfit(M, I_truth/255, order);
inv_val = polyval(I_poly, M);
figure
plot(M, inv_val)
hold on
plot(M, I_truth/255)

%% polynomial fit to crf => intensity
% input is irradiances I
% output is intensities M 

hold on
M_poly = polyfit(I_truth/255, M, order);
crf_val = polyval(M_poly, I_truth/255);
plot(I_truth/255, crf_val)
hold on
plot(I_truth/255, M) % 
hold off

%% store the exposure 
exposurelist=dlmread('C:\Users\User\Desktop\AUB\spring 2022\dataset\calib_narrowGamma_sweep3\times.txt' );
exposures=exposurelist(:,3);

%% simulation
% first image
% use I_poly and M_poly
% K = 0.99;
% M1 = 0.1:0.001:0.9;
% net=denoisingNetwork('DnCNN');
image2= imread('C:\Users\User\Desktop\AUB\spring 2022\dataset\calib_narrowGamma_sweep3\00546.jpg');
% image2_ = im2double(image2_);
%image2 = medfilt2(image2,[50 50]);
% image2=imgaussfilt(image2_,1);
% image2=denoiseImage(image2_,net);
% montage({image2_,image2});
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

% Find magnitude and orientation of gradient
mag=uint8(round( sqrt(Ix.^2 + Iy.^2) ));
mag=double(mag(:));

image1= imread('C:\Users\User\Desktop\AUB\spring 2022\dataset\calib_narrowGamma_sweep3\00539.jpg');
% image1=im2double(image1);
%image1 = medfilt2(1);
%image(image1)
image1_exposure=exposures(539 + 1);
image2_exposure=exposures(546 + 1);

M2 = image2(:);
K = image1_exposure/image2_exposure;%0.9522749266;%1.0501169;%0.9522749266; % should lead me to 00546
temp = polyval(I_poly, double(M2)) * K;
M1 = double(polyval(M_poly, temp));

douaa_wrong=reshape(M1,[size(image2,1) size(image2,2)]);

%% system
corr_pts = size(M1,2);
counter = 1;
r = K;
for corr_pt=1:50:corr_pts

    [ M1_pixel , M2_pixel ] = deal( M1(corr_pt), M2(corr_pt) ) ;

    % no weight 
    row=(1/mag(corr_pt))*[ (1 - r)  (M1_pixel - r*M2_pixel)  (M1_pixel^2- r*M2_pixel^2)   (M1_pixel^3- r*M2_pixel^3)  (M1_pixel^4- r*M2_pixel^4)    (M1_pixel^5- r*M2_pixel^5)   (M1_pixel^6- r*M2_pixel^6)   (M1_pixel^7- r*M2_pixel^7)    (M1_pixel^8- r*M2_pixel^8)   (M1_pixel^9- r*M2_pixel^9)  (M1_pixel^10- r*M2_pixel^10)];

    A(counter,:)=row;

    counter= counter+1;

end




%% solve
A=double(A);
% n=11;
% cvx_begin
% 
%     variable x(n)
%     %b=ones(corr_pts,1)
%     %minimize( norm(A*x - b) );
%     minimize( (A*x)'*(A*x) );
%     subject to 
%     %sum(x)==1;
%     %x(1)==0;
%     %sum(x)== 1;
%     x(1) + x(2)*255 +  x(3)* 255^2 + x(4)*255^3 +  x(5)* 255^4 + x(6)*255^5 +  x(7)* 255^6 + x(8)*255^7 +  x(9)* 255^8 +  x(10)* 255^9 +  x(11)* 255^10 == 255;
%  
%   
% cvx_end
% 
corr_mat = A'*A;
[U, S, V] = svd(A);
x = V(:, 11); % this should be compared to I_truth


%% plot 
%plot 
%input=0:1/255:1; 
input=0:1:255;

final= x(1) + x(2)*255 +  x(3)* 255^2 + x(4)*255^3 +  x(5)* 255^4 + x(6)*255^5 +  x(7)* 255^6 + x(8)*255^7 +  x(9)* 255^8 +  x(10)* 255^9 +  x(11)* 255^10;
% final=sum(x);
%ratio=1/final;
ratio=255/final;

output= x(1) + x(2)*input +  x(3)*input.^2  + x(4)*input.^3 +  x(5)* input.^4 + x(6)*input.^5 +  x(7)* input.^6 + x(8)*input.^7 +  x(9)* input.^8 +  x(10)* input.^9 +  x(11)* input.^10;

figure('Name','j')
%plot(input,output)
plot(input,output*ratio)
hold on
%plot(input,crf_truth/255)
plot(input,I_truth)