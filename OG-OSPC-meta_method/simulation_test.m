clc
clear all


%% read inv crf truth
I_truth=dlmread('C:\Users\User\Desktop\AUB\spring 2022\dataset\calib_narrowGamma_sweep3\pcalib.txt' );

%% polynomial fit to inv crf => irradiance
% input is intensities M 
% output is irradiances I
M = 0:1/255:1;
figure
order = 15;
I_poly = polyfit(M, I_truth/255, order);
inv_val = polyval(I_poly, M);
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
plot(I_truth/255, M)  
hold off

%% simulation
% first image
% use I_poly and M_poly
K =0.5;

M2 = 0:0.001:1;
%M2=[M2  ]
temp = polyval(I_poly, M2) * K;
M1 = polyval(M_poly, temp) ;
M1=[ M1(1:500)+0.01  M1(501:size(M2,2))+0.01    ];

%% system
corr_pts = size(M1,2);
counter = 1;
r = K;
for corr_pt=1:1:corr_pts

    [ M1_pixel , M2_pixel ] = deal( M1(corr_pt), M2(corr_pt) ) ;

    % no weight 
    row=[ (1 - r)  (M1_pixel - r*M2_pixel)  (M1_pixel^2- r*M2_pixel^2)   (M1_pixel^3- r*M2_pixel^3)  (M1_pixel^4- r*M2_pixel^4) (M1_pixel^5- r*M2_pixel^5) (M1_pixel^6- r*M2_pixel^6) (M1_pixel^7- r*M2_pixel^7) (M1_pixel^8- r*M2_pixel^8) (M1_pixel^9- r*M2_pixel^9) (M1_pixel^10- r*M2_pixel^10)];

    A(counter,:)=row;

    counter= counter+1;

end

%% solve

n=11;
cvx_begin

    variable x(n)
    %b=ones(corr_pts,1)
    %minimize( norm(A*x - b) );
    minimize( (A*x)'*(A*x) );
    subject to 
    %sum(x)==1;
    %x(1)==0;
    sum(x)== 1;
    %x(1) + x(2)*255 +  x(3)* 255^2  == 255; 
cvx_end

%corr_mat = A'*A;
% [U, S, V] = svd(A);
% x = V(:, 11); % this should be compared to I_truth

%% plot

input=0:1/255:1;

output =  x(1) + x(2)*input +  x(3)* input.^2 + x(4)*input.^3 +  x(5)* input.^4 + x(6)*input.^5 +  x(7)* input.^6 + x(8)*input.^7 +  x(9)* input.^8 +  x(10)* input.^9 +  x(11)* input.^10;
%scale = x(1) + x(2) + x(3)+x(4) + x(5) + x(6)+ x(7)+x(8) + x(9) + x(10) + x(11);
scale = sum(x);

%plot(input,output)
figure
plot(input, output)
hold on

plot(input, I_truth/255)


%% the far I go from r=1 the better result I get
%% 10 points gave a good estimate with poly of 11 unkowns 
%% with one pair (one r we could get good estimate )
%% having many ones did not affect the estimation 
