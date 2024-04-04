clc
clear all
M1_all=[];
%% store the exposure 
M=dlmread('C:\Users\User\Desktop\AUB\spring 2022\dataset\calib_narrowGamma_sweep3\times.txt' );
exposures=M(:,3);

%% read crf
crf_truth=dlmread('C:\Users\User\Desktop\AUB\spring 2022\dataset\calib_narrowGamma_sweep3\pcalib.txt' );


sum1=zeros(1024,1280);
sum2=zeros(1024,1280);
%% read pair one  images with same e  and average
%image1_list=820:1:827;
image1_list=910:1:921;
for pair=1:1:size(image1_list,2)
    
%% read  2 images 
image1=imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\calib_narrowGamma_sweep3\00" + num2str(image1_list(pair)) + ".jpg");

%% 800-890 900-990
%% normalize images

image1=im2double(image1); %./ max(image1(:));
sum1=image1 + sum1;

end
image1_avg=sum1/size(image1_list,2);

%% read the second pair of images with smae e and average
%image2_list=828:1:834;
image2_list=922:1:930;
for pair=1:1:size(image2_list,2)
    
%% read  2 images 

image2=imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\calib_narrowGamma_sweep3\00" + num2str(image2_list(pair)) + ".jpg");

%% 800-890 900-990
%% normalize images

image2=im2double(image2); 
sum2=image2 + sum2;
end
image2_avg=sum2/size(image2_list,2);






image1= image1_avg;
image2=image2_avg;
r_all=[];
A_all=[];

%% solve

image1_exposure=exposures(921 + 1);
image2_exposure=exposures(922 + 1);

r= (image1_exposure/image2_exposure);
r_all=[r_all r];
%r= 0.93;
total_size=size(image1,1)*size(image1,2);
% figure('Name','image 1')
% imshow(image1)
% figure('Name','image 2')
% imshow(image2)





%% flat
image1_flat=reshape(double(image1),[1 total_size]);   %note
image2_flat=reshape(double(image2),[1 total_size ]);

%% fill the array 

corr_pts=total_size;

[Y,E] = discretize(image1_flat ,10);
counter=1;

if r~=1
    
for corr_pt=1:500:corr_pts

    [ M1 , M2]=deal( double(image1_flat(corr_pt)), double(image2_flat(corr_pt))) ;
    
    %%%% found a mistake 
    %if M1 == 1 || M2 ==1 || M1 == 0 || M2 == 0 || M1>M2 
    
    if  M1 == 1  || M1==M2 || M1>M2 
        
    else
        
        M1_all=[ M1_all M1];   
        %% no weight 
        %(1/mag(corr_pt))*
        row=[ (1 - r)    (M1 - r*M2)   (M1^2- r*M2^2) (M1^3- r*M2^3) (M1^4- r*M2^4)  (M1^5- r*M2^5) (M1^6- r*M2^6) (M1^7- r*M2^7) (M1^8- r*M2^8) (M1^9- r*M2^9) (M1^10- r*M2^10)];
        A(counter,:)=row;
        %F(corr_pt)= mult;
        counter=counter+1;

    end

end


A_all=[A_all ; A];
A=[];

end



%% solve 

n=11;
cvx_begin

    variable x(n)
    %b=ones(corr_pts,1)
    %minimize( norm(A*x - b) );
    minimize( (A_all*x)'*(A_all*x) );
    
    subject to 
    sum(x)==1;
    x(1)==0;
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
plot(input,output*ratio,'--','LineWidth',3)
hold on
plot(input,crf_truth/255,'--','LineWidth',3)
% plot(input,crf_truth)

%% residual 

res=A_all*x;
RSE=(res)'*(res);
RMSE=sqrt(  (RSE/size(A_all,1)) ) ;

res_t= (output - (crf_truth/255))';
RSE_t=res_t'*res_t;
RMSE_t=sqrt(  (RSE_t/size(A_all,1)) ) ;









