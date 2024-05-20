%% load
clear all
clc

B=load('C:\Users\User\Desktop\AUB\spring 2022\codes\meta method\data\B.mat');

g0=load('C:\Users\User\Desktop\AUB\spring 2022\codes\meta method\data\g0.mat');

hinv1=load('C:\Users\User\Desktop\AUB\spring 2022\codes\meta method\data\hinv1.mat');
hinv2=load('C:\Users\User\Desktop\AUB\spring 2022\codes\meta method\data\hinv2.mat');
hinv3=load('C:\Users\User\Desktop\AUB\spring 2022\codes\meta method\data\hinv3.mat');
hinv4=load('C:\Users\User\Desktop\AUB\spring 2022\codes\meta method\data\hinv4.mat');

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

%% LET US START





%% read crf
crf_truth=dlmread('C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_30\pcalib.txt' );

% figure(1)
% plot(crf_truth)


%% get corresponding points %0.95 0.9376 0.9376

num_image1="00185"; %105 1  35 185 1585
num_image2="00186"; %106 36 36 186 1586
%distance below 10  as lower bound, still works at d 250 
I1 = im2double(imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_30\images\" + num_image1 + ".jpg" )); %0 102
I2 = im2double(imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_30\images\" + num_image2 + ".jpg" )); %15 128


%I1 = rgb2gray (im2double(imread('C:\Users\User\Desktop\102_m.jpg'))); 
%I2 = rgb2gray (im2double(imread('C:\Users\User\Desktop\108_m.jpg'))); 
% figure
% imshow(I1)
% figure
% imshow(I2)

%% exposures
temp_exposures = dlmread('C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_30\times.txt' );
exposures_truth = temp_exposures(:,3);
image1_exposure = exposures_truth(str2double(num_image1) + 1);
image2_exposure = exposures_truth(str2double(num_image2) + 1);

% exposure ratio
exposure_ratio = (image1_exposure/image2_exposure);


%% feature extraction  and matching 
feature_method="sift"
tol=20;
if feature_method=="sift"
    N_corr=500; % deleted it in the function , it is taking them all 
    draw=0;
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
   
    matchedPointsf1_r=matchedPointsf1_r(dist<tol,:);
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

%% filling
%try r estimate 
r=exposure_ratio;
%% bins 
[Y,E] = discretize(M_f1,10);   

counter=1;
for corr_pt =  1:corr_pts
    
% each iter adds one equation from one pair of corresponding points    
[M1,M2 ] = deal(double(M_f1(corr_pt)), double(M_f2(corr_pt)));
bin_number=Y(corr_pt);
length_bin=sum(Y==bin_number);

    if M1 ~= 1 || M2 ~=1 || M1 ~= 0 || M2 ~= 0
        h1_M1=polyval(hinv1_fn,M1);
        h1_M2=polyval(hinv1_fn,M2);
        
        h2_M1=polyval(hinv2_fn,M1);
        h2_M2=polyval(hinv2_fn,M2);
        
        h3_M1=polyval(hinv3_fn,M1);
        h3_M2=polyval(hinv3_fn,M2);
        
%         h4_M1=polyval(hinv4_fn,M1);
%         h4_M2=polyval(hinv4_fn,M2);
        
        g0_M1=polyval(g0_fn,M1);
        g0_M2=polyval(g0_fn,M2);
        
 
        
        %% no weight 
        row = [ (h1_M1 - r*h1_M2) (h2_M1 - r*h2_M2) (h3_M1 - r*h3_M2)];% (h4_M1 - r*h4_M2)  ];    %(M1^3- r*M2^3)  (M1^4- r*M2^4)]; %(M1^5- r*M2^5) (M1^6- r*M2^6) (M1^7- r*M2^7) (M1^8- r*M2^8) (M1^9- r*M2^9)];
        %row=(1/length_bin)*[ (1 - r) (M1 - r*M2)   (M1^2- r*M2^2)  ];
        element= -g0_M1 + r*g0_M2;
        
        A(counter,:)=row;
        b(counter,1)=element;
        %F(corr_pt)= mult;
        counter=counter+1;

    end

end

%% solve 

n=3;
cvx_begin

    variable x(n)
    %b=ones(corr_pts,1)
    %minimize( norm(A*x - b) );
    minimize( (A*x - b )'*(A*x - b) );
       
cvx_end

%% using pesudo inverse 
cs=inv(A'*A)*A'*b;


%% plot 
%plot 
input=0:1/255:1;

h1_M=polyval(hinv1_fn,input);
h2_M=polyval(hinv2_fn,input);
h3_M=polyval(hinv3_fn,input);
% h4_M=polyval(hinv4_fn,input);
g0_M=polyval(g0_fn,input);

finv= g0_M + x(1)*h1_M + x(2)*h2_M + x(3)*h3_M %+ x(4)*h4_M ;

figure('Name','j')
plot(input,finv)
hold on

plot(input,crf_truth/255)


%% polyfit 

order=4
p = polyfit(input,crf_truth/255,order);
f1 = polyval(p,input);
plot(input,f1)
hold on
plot(input,crf_truth/255)
























