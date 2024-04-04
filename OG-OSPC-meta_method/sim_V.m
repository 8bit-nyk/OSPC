clc
clear all


%% read inv crf truth
I_truth=dlmread('C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_50\pcalib.txt' );

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

%% V fit 


vignette_truth = imread('C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_50\vignette.png');

vignette_truth_norm = im2double(vignette_truth);


%get radius for  image
% calculate a radius map

I1=im2double(imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_50\images\00001.jpg"));
radius_map_f1 = zeros(size(I1));
center_I1 = size(I1)/2;
for i = 1 :1:size(I1,1)
    for j = 1:1:size(I1, 2)
        
        radius_map_f1(i,j) = sqrt(   (i - center_I1(1))^2 + (j - center_I1(2))^2 );       
    end
end

radius_map_f1=radius_map_f1 /  max(radius_map_f1, [], 'all'); %% why normalize?!




eps = 0.005;
radius_map_norm_sub = radius_map_f1(center_I1(1):size(I1,1), center_I1(2):size(I1,2));
vignette_truth_norm_sub = vignette_truth_norm(center_I1(1):size(I1,1), center_I1(2):size(I1,2));

rad_counter = 0;
for rad = 0:radius_map_f1(1,1)/1000:radius_map_f1(1,1)
    [rad_row, rad_col] = find(radius_map_norm_sub < (rad + eps) & radius_map_norm_sub > (rad - eps), 1);
    if size(rad_row) ~= 0 & size(rad_col) ~=0
        rad_counter = rad_counter + 1;
        radius_input(1,rad_counter) = radius_map_norm_sub(rad_row, rad_col);
        vignette_output(1, rad_counter) = vignette_truth_norm_sub(rad_row, rad_col);
        plot(radius_input(1, rad_counter), vignette_output(1, rad_counter), '+');
        hold on
    end
end

vig_pol_order = 4;
vign_truth_fun = polyfit(radius_input, vignette_output, vig_pol_order);
vign_truth_eval = polyval(vign_truth_fun, radius_input);
plot(radius_input, vign_truth_eval)
hold on
scatter(0,0)


%% start simulation 
M1_list=[]

R1_list= [ 0:0.00001:0.3  ]  ;

% j=0;
% while(j<3)
% R1_list=[ R1_list  R1_list]
% j=j+1;
% end

R1_list=[ R1_list 0.3:0.001:0.8 ] ;
%0:0.001:1;
R2_list= R1_list + 0.2  ; 
M2_list= 0.5*ones(1,size(R1_list,2)) ; % 0:0.001:0.8 %

r=0.9;
Vf1=polyval(vign_truth_fun, R1_list );
Vf2=polyval(vign_truth_fun, R2_list );
invf2=polyval(I_poly,  M2_list );
M1_list=polyval(M_poly, r*(Vf1./Vf2).* invf2 );

corr_pts=size(M1_list,2);
 noise=0 %0.01 % -0.04 %.02 ;
 
%% estimate 
[Y,E] = discretize(R1_list,0:0.1:1);
counter=1;

for i=1:corr_pts
    
[ M1 , M2,R1,R2]=deal( double(M1_list(i)), double(M2_list(i)),double(R1_list(i)),double(R2_list(i)) );

bin_number=Y(i);
length_bin=sum(Y==bin_number);

invf1=polyval(I_poly,M1)    ; %%% changed 
invf2=polyval(I_poly,M2 + noise) ;


J=(invf1/invf2)*(1/r); 
a = R1^2 - J * R2^2;
b = R1^4 - J * R2^4;
c = R1^6 - J * R2^6;


%(1/mag2(i))*
%(1/length_bin)*
A(counter,:)=[ a b c]; % add 1/gradient  % a smart idea to draw A each row as vectors
 
l(counter,:)=  ( J - 1 ) ;
counter=counter + 1;

end


% A_all=[ A_all ; A ]; 
% l_all=[ l_all ; l ];


A_all=A;
l_all=l;

n=3;
cvx_begin
    variable x(n)
   
    %minimize( norm(A*x - l) ); %try with and without square
    %minimize( power(norm(A*x ),2) )
    %minimize(power(2,norm(A*x,2)))
    minimize( (A_all*x -l_all)'*(A_all*x - l_all) )
% with constraint 
    %subject to 
    %sum(x) == 1;
cvx_end

%evector_min=x;
[Vhomo,D]=eig(A_all'*A_all);
%% pesudo method 
V = inv(A_all'*A_all)*A_all'*l_all;

%% plot vignette
vignette_image = zeros(size(I1));
for i = 1:1:size(vignette_image,1)
    for j = 1:1:size(vignette_image, 2) 

       temp=1 + V(1)*radius_map_f1(i,j)^2 + V(2)*radius_map_f1(i,j)^4 + V(3)*radius_map_f1(i,j)^6;
       
       if temp  >= 0 
       %vignette_image(i,j) = 1 -6.597*radius_map_f1(i,j)^2 + 14.17*radius_map_f1(i,j)^4 + -9.75*radius_map_f1(i,j)^6;       
       vignette_image(i,j) = temp;   
        end
    end
end 

figure(3)

imshow(vignette_image)
figure
% in= 0:0.1:1 ;
in= 0:radius_map_f1(1,1)/1000:radius_map_f1(1,1) ;
plot( in, 1 + V(1)*in.^2 +  V(2)*in.^4 + V(3)*in.^6 );
figure
plot( in, 1 + x(1)*in.^2 +  x(2)*in.^4 + x(3)*in.^6 );
hold on 

plot(radius_input, vign_truth_eval,'r')
hold on
scatter(0,0)


% 0 and one bil M naza3ta 

