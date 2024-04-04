%% loop pairs
%%% read image pair 
%%% extract features 
%%% fill array 
%% end loop 
%% solve 


clc
clear all


%% store the exposure 
M=dlmread('C:\Users\User\Desktop\AUB\spring 2022\dataset\calib_narrowGamma_sweep3\times.txt');
exposures=M(:,3);

%% read crf
crf_truth=dlmread('C:\Users\User\Desktop\AUB\spring 2022\dataset\calib_narrowGamma_sweep3\pcalib.txt' );

figure(1)
plot(crf_truth)
A_all=[];
b_all=[];
M1_all=[];
r_all=[];

start_=800;
end_=950;

image1_list=start_:1:end_;
image2_list=start_ + 1:1:end_ + 1;

image1_list=[image1_list 415:1:520];
image2_list=[image2_list 416:1:521];
% 
% image1_list=[900          ];
% image2_list=[910         ];
%% loop start 
for pair=1:1:size(image1_list,2)
    
%% read  image pair  
if image1_list(pair) >99 && image1_list(pair) < 1000
image1=imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\calib_narrowGamma_sweep3\00" + num2str(image1_list(pair)) + ".jpg");
image2=imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\calib_narrowGamma_sweep3\00" + num2str(image2_list(pair)) + ".jpg");
elseif image1_list(pair) < 100 
image1=imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\calib_narrowGamma_sweep3\000" + num2str(image1_list(pair)) + ".jpg");
image2=imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\calib_narrowGamma_sweep3\000" + num2str(image2_list(pair)) + ".jpg");
elseif image1_list(pair) > 999
image1=imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\calib_narrowGamma_sweep3\0" + num2str(image1_list(pair)) + ".jpg");
image2=imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\calib_narrowGamma_sweep3\0" + num2str(image2_list(pair)) + ".jpg");    
end

%% 800-890 900-990
%% normalize images

image1=im2double(image1); %./ max(image1(:));
image2=im2double(image2); %./ max(image2(:));
I1=image1;
I2=image2;

%% filter 
% image1 = medfilt2(image1,[3 3]);
% image2 = medfilt2(image2, [3 3]);
% image2=image2 + a/2;
% image1=image1 + a/2;

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

%% mag

kx=[-1,0,1;
    -2,0,2;
    -1,0,1] ;

ky=[-1,-2,-1;
    0,0,0;
    1,2,1] ;

%Inorm = conv2(double(I), norm, 'same');
%Inorm=uint8(round(Inorm))
%imshow(Inorm)
%% find features 
%% feature extraction  and matching 
feature_method="sift"
tol=8;
draw=0;
if feature_method=="sift"
    N_corr=500; % deleted it in the function , it is taking them all 
    
    tic
    [X_image1,X_image2,limit]=meta_sift_corr_fn_photo(I1,I2,N_corr,draw);
    toc
    % calculate distance
    x_sub_sq=( X_image1(1,:)- X_image2(1,:) ).^2;
    y_sub_sq=( X_image1(2,:)- X_image2(2,:) ).^2;
    d= x_sub_sq + y_sub_sq ;
    dist= (d).^0.5;
    %% tol 
    X_image1=X_image1(:,dist<tol);
    X_image2=X_image2(:,dist<tol);
    figure ; clf ;
    imagesc(cat(2, I1, I2)) ;


    xa = X_image1(1,:) ;
    xb = X_image2(1,:) + size(I1,2) ;
    ya = X_image1(2,:) ;
    yb = X_image2(2,:) ;
    
    hold on ;
    h = line([xa ; xb], [ya ; yb]) ;
    set(h,'linewidth', 1, 'color', 'r') ;
    
    
    num_corr_points=size(X_image1(1,:),2);
    for corr=1:num_corr_points

        M_f1(1,corr)=I1(round(X_image1(2,corr)),round(X_image1(1,corr))); %% check why inverted since y,x row col
        M_f2(1,corr)=I2(round(X_image2(2,corr)),round(X_image2(1,corr)));

    end

else 
 %% using other feature detectors
% 

    tic

    % sift 
%     points1_ = detectSIFTFeatures(I1);
%     points2_ = detectSIFTFeatures(I2);
    %harris
%     points1_ = detectHarrisFeatures(I1);
%     points2_ = detectHarrisFeatures(I2);
%    surf
%     points1_ = detectSURFFeatures(I1);
%     points2_ = detectSURFFeatures(I2);

    % orb
    points1_ =detectORBFeatures(I1);
    points2_ =detectORBFeatures(I2);

    % Brisk
%     points1_ = detectBRISKFeatures(I1);
%     points2_ = detectBRISKFeatures(I2);
    % Fast 
%     points1_ = detectFASTFeatures(I1);
%     points2_ = detectFASTFeatures(I2);


    % strong 
    % N=3000;
    % points1 = selectStrongest(points1_,N);
    % points2 = selectStrongest(points2_,N);
%    Extract the features
    [f1,vpts1] = extractFeatures(I1,points1_);
    [f2,vpts2] = extractFeatures(I2,points2_);
    %Retrieve the locations of matched points

    indexPairs = matchFeatures(f1,f2) ;

    matchedPoints1 = vpts1(indexPairs(:,1));
    matchedPoints2 = vpts2(indexPairs(:,2));
    %display

    figure; showMatchedFeatures(I1,I2,matchedPoints1,matchedPoints2);
    legend('matched points 1','matched points 2');

    matchedPointsf1_r=round(matchedPoints1.Location);
    matchedPointsf2_r=round(matchedPoints2.Location);
    toc
    %% calculate distance
    x_sub_sq=( matchedPointsf1_r(:,1)- matchedPointsf2_r(:,1) ).^2;
    y_sub_sq=( matchedPointsf1_r(:,2)- matchedPointsf2_r(:,2) ).^2;
    d= x_sub_sq + y_sub_sq ;
    dist= (d).^0.5;
   
    matchedPointsf1_r=matchedPointsf1_r(dist<tol,:);  %% choose points with small disp 
    matchedPointsf2_r=matchedPointsf2_r(dist<tol,:);
    % draw 
    figure(2) ; clf ;
    imagesc(cat(2, I1, I2)) ;

    xa = matchedPointsf1_r(:,1)' ;
    xb = matchedPointsf2_r(:,1)' + size(I1,2) ;
    ya = matchedPointsf1_r(:,2)' ;
    yb = matchedPointsf2_r(:,2)' ;

    hold on ;
    h = line([xa ; xb], [ya ; yb]) ;
    set(h,'linewidth', 1, 'color', 'b') ;

    num_corr_points=size(matchedPointsf1_r(:,1),1);


    for corr=1:num_corr_points 

        M_f1(1,corr)=I1(matchedPointsf1_r(corr,2),matchedPointsf1_r(corr,1)); %% check why inverted 
        M_f2(1,corr)=I2(matchedPointsf2_r(corr,2),matchedPointsf2_r(corr,1));

    end


    num_corr_points=size(M_f1,2);
end

%% filling 
corr_pts=num_corr_points; % changing this drastically changes the result thus outliers play a major role , so remove them



counter=1;
for corr_pt =  1:corr_pts
    
% each iter adds one equation from one pair of corresponding points    
[M1,M2 ] = deal(double(M_f1(corr_pt)), double(M_f2(corr_pt)));

    %% found a mistake
     %if M1 ~= 1 || M2 ~=1 || M1 ~= 0 || M2 ~= 0
     if  M1 == 1 || M2 == 1 ||M1 == 0 || M2 == 0 || M1==M2 || M1>M2
         continue 
     else
        %% no weight 
        row=[ (1 - r) (M1 - r*M2)   (M1^2- r*M2^2) (M1^3- r*M2^3) (M1^4- r*M2^4) (M1^5- r*M2^5) (M1^6- r*M2^6) (M1^7- r*M2^7) (M1^8- r*M2^8) (M1^9- r*M2^9) (M1^10- r*M2^10)];
        %% intensity weight 
        % if bin_number==1
        %row=(1/length_bin)*[ (1 - r) (M1 - r*M2)   (M1^2- r*M2^2)  ];
        % else
        %row=(1/length_bin)*[ (1 - r) (M1 - r*M2)   (M1^2- r*M2^2)  ];
        % end
        %% intensity exponential 
        %row=(1/exp(length_bin))*[ (1 - r) (M1 - r*M2)   (M1^2- r*M2^2)  ];    %(M1^3- r*M2^3)  (M1^4- r*M2^4)]; %(M1^5- r*M2^5) (M1^6- r*M2^6) (M1^7- r*M2^7) (M1^8- r*M2^8) (M1^9- r*M2^9)];

        %% idistance + intensity weight 

        %row=(1/length_bin)*double((1/dist(corr_pt)))*[ (1 - r) (M1 - r*M2)   (M1^2- r*M2^2)  ];    %(M1^3- r*M2^3)  (M1^4- r*M2^4)]; %(M1^5- r*M2^5) (M1^6- r*M2^6) (M1^7- r*M2^7) (M1^8- r*M2^8) (M1^9- r*M2^9)];
        %row=( 1/( length_bin + dist(corr_pts) ))*[ (1 - r) (M1 - r*M2)   (M1^2- r*M2^2)  ];    %(M1^3- r*M2^3)  (M1^4- r*M2^4)]; %(M1^5- r*M2^5) (M1^6- r*M2^6) (M1^7- r*M2^7) (M1^8- r*M2^8) (M1^9- r*M2^9)];

        A(counter,:)=row;
        %F(corr_pt)= mult;
        k=1;
        counter=counter + 1;
      end
end

A_all=[A_all ; A];
A=[];

end

%% loop end 
%% solve 

n=11;
cvx_begin

    variable x(n)
    %b=ones(corr_pts,1)
    %minimize( norm(A*x - b) );
    minimize( (A_all*x)'*(A_all*x) );
    
    subject to 
    sum(x)==1;
    %x(1)==0;
    %x(1) + x(2)*0.5725 +  x(3)* 0.5725^2 + x(4)*0.5725^3 +  x(5)*0.5725^4 + x(6)*0.5725^5 +  x(7)*0.5725^6 + x(8)*0.5725^7 +  x(9)*0.5725^8 +  x(10)*0.5725^9 +  x(11)*0.5725^10 == 0.2971;

    %x(1) + x(2)*255 +  x(3)* 255^2 + x(4)*255^3 +  x(5)* 255^4 + x(6)*255^5 +  x(7)* 255^6 + x(8)*255^7 +  x(9)* 255^8 +  x(10)* 255^9 +  x(11)* 255^10 == 255;
    %x(1) + x(2)*255 +  x(3)* 255^2  == 255; 
     
       
cvx_end

%% eigen method 
[V,D]=eig(A_all'*A_all);

% [U,S,V]=svd(A_all);  
% x=V(:,1);

%% plot 
%plot 
input=0:1/255:1; 
%input=0:1:255;

%final= x(1) + x(2)*255 +  x(3)* 255^2 + x(4)*255^3 +  x(5)* 255^4 + x(6)*255^5 +  x(7)* 255^6 + x(8)*255^7 +  x(9)* 255^8 +  x(10)* 255^9 +  x(11)* 255^10;
final=sum(x);
ratio=1/final;
%ratio=255/final;
output=  x(1) + x(2)*input +  x(3)*input.^2  + x(4)*input.^3 +  x(5)* input.^4 + x(6)*input.^5 +  x(7)* input.^6 + x(8)*input.^7 +  x(9)* input.^8 +  x(10)* input.^9 +  x(11)* input.^10;
figure('Name','j')
%plot(input,output)
plot(input,output*ratio,'--')
hold on
plot(input,crf_truth/255,'--')
% plot(input,crf_truth)

%% residual 

res=A_all*x;
RSE=(res)'*(res);
RMSE=sqrt(  (RSE/size(A_all,1)) ) ;

res_t= (output - (crf_truth/255))';
RSE_t=res_t'*res_t;
RMSE_t=sqrt(  (RSE_t/size(A_all,1)) ) ;



