clc
clear all


%% read images with same exposure 

fixed  = im2double(imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_50\images\00086.jpg" )); %0 102
moving = im2double(imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_50\images\00096.jpg" )); 67
%% exposures
temp_exposures = dlmread('C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_50\times.txt' );
exposures_truth = temp_exposures(:,3);
image1_exposure = exposures_truth(86 + 1);                  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
image2_exposure = exposures_truth(96 + 1);

% exposure ratio
exposure_ratio = (image1_exposure/image2_exposure);
%% register 
registrationEstimator (moving,fixed);  
%% subtract 
reg=movingReg.RegisteredImage ;
sub1=fixed - reg ;

reg_nopad=  reg(reg(:)~=0)  ; 
sub1_nopad= sub1(reg(:)~=0) ;
%% get M1 and M2
image1_flat=fixed(:);
image1_flat_nopad=image1_flat( reg(:) ~= 0);

% M1_list=image1_flat_nopad(abs(sub1_nopad) < 0.0005);
% M2_list=reg_nopad(abs(sub1_nopad) < 0.0005);
c=1;
for i=1:10:size(reg_nopad,1)
    
M1_list(c,1)=image1_flat_nopad(i,1);
M2_list(c,1)=reg_nopad(i,1);
c=c+1;
end

corr_pts=size(M1_list,1);
counter=1;
r=exposure_ratio;

I1=fixed;
%% estimate V
% get raduis 
% radius_map_f1 = zeros(size(I1));
% center_I1 = size(I1)/2;
% for i = 1 :1:size(I1,1)
%     for j = 1:1:size(I1, 2)
%         
%         radius_map_f1(i,j) = sqrt(   (i - center_I1(1))^2 + (j - center_I1(2))^2 );       
%     end
% end
% 
% radius_map_f1=radius_map_f1; %/ max(radius_map_f1, [], 'all'); %% why normalize?!
% 
% raduis_map_f1_flat=radius_map_f1(:);
% raduis_map_f1_flat_nopad=raduis_map_f1_flat(reg ~= 0)





%% estimate crf 
for corr_pt =  1:corr_pts
    
% each iter adds one equation from one pair of corresponding points    
[M1,M2 ] = deal(double(M1_list(corr_pt)), double(M2_list(corr_pt)));

    %% found a mistake
     %if M1 ~= 1 || M2 ~=1 || M1 ~= 0 || M2 ~= 0
%      if  M1 == 1 || M2 == 1 ||M1 == 0 || M2 == 0 || M1==M2 || M1>M2 %|| R_f1(corr_pt) > 0.1
%          continue 
%      else
        %% no weight 
        row=[ (1 - r) (M1 - r*M2)   (M1^2- r*M2^2)];%  (M1^3- r*M2^3) (M1^4- r*M2^4) (M1^5- r*M2^5) (M1^6- r*M2^6) (M1^7- r*M2^7) (M1^8- r*M2^8) (M1^9- r*M2^9) (M1^10- r*M2^10)];
        %M1_all=[M1_all M1];
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






%% loop end 
%% solve 

n=3;
cvx_begin

    variable x(n)
    %b=ones(corr_pts,1)
    %minimize( norm(A*x - b) );
    minimize( (A*x)'*(A*x) );
    
    subject to 
    sum(x)==1;
   % x(1)==0;
    %x(1) + x(2)*0.5725 +  x(3)* 0.5725^2 + x(4)*0.5725^3 +  x(5)*0.5725^4 + x(6)*0.5725^5 +  x(7)*0.5725^6 + x(8)*0.5725^7 +  x(9)*0.5725^8 +  x(10)*0.5725^9 +  x(11)*0.5725^10 == 0.2971;

    %x(1) + x(2)*255 +  x(3)* 255^2 + x(4)*255^3 +  x(5)* 255^4 + x(6)*255^5 +  x(7)* 255^6 + x(8)*255^7 +  x(9)* 255^8 +  x(10)* 255^9 +  x(11)* 255^10 == 255;
    %x(1) + x(2)*255 +  x(3)* 255^2  == 255; 
     
       
cvx_end

%% eigen method 
[V,D]=eig(A'*A);

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
output=  x(1) + x(2)*input +  x(3)*input.^2 %+ x(4)*input.^3 +  x(5)* input.^4 + x(6)*input.^5 +  x(7)* input.^6 + x(8)*input.^7 +  x(9)* input.^8 +  x(10)* input.^9 +  x(11)* input.^10;
figure('Name','j')
%plot(input,output)
plot(input,output*ratio,'--','LineWidth',3)






