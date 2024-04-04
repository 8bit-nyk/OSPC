counter=1;
bo=L_per_ss>99 & L_per_ss<101;
M_f1=M_f1(bo)
M_f2=M_f2(bo)
 
R_f1= R_f1(bo)
R_f2= R_f2(bo)




corr_pts=size(M_f1,2); % changing this drastically changes the result thus outliers play a major role , so remove them

%num_corr_points


for i=1:corr_pts
    
[ M1 , M2,R1,R2]=deal( double(M_f1(i)), double(M_f2(i)),double(R_f1(i)),double(R_f2(i)) );
%[c0,c1,c2]=deal(0,-0.093920939562920,1.093920939562920); %%% check tihs 255 , 1 this is only for 0 to 1

%invf1= c0 + M1*c1 + M1^2*c2; %+ M1^3*X(4) + M1^4*X(5) + M1^5*X(6) + M1^6*X(7) + M1^7*X(8) + M1^8*X(9) + M1^9*X(10); 
%invf2= c0 + M2*c1 + M2^2*c2;
invf1=polyval(p,M1)    ;
invf2=polyval(p,M2)    ;


R1_list=[R1_list R1];


% I flipped R1 and R2 by mistake
% a = (invf1/invf2) * R2^2 - exposure_ratio * R1^2;
% b = (invf1/invf2) * R2^4 - exposure_ratio * R1^4;
% c = (invf1/invf2) * R2^6 - exposure_ratio * R1^6;
% a =  R2^2 -  R1^2  ;
% b =  R2^4 -  R1^4  ;
% c =  R2^6 -  R1^6  ;


J=(invf1*image2_exposure)/(invf2*image1_exposure); 
a = R1^2 - J * R2^2;
b = R1^4 - J * R2^4;
c = R1^6 - J * R2^6;

test_arr(counter)= M2 - M1 ;
test_arr1(counter)=dist(i);
test_arr2(counter)=R2-R1;  %% why different than dist ?
%(1/mag2(i))*
A1(counter,:)=[ a b c]; % add 1/gradient  
% l(counter,:)= (- (invf1/invf2) + exposure_ratio) ; % add 1/gradient  
% l1(counter,:)= 1 - ( (invf1*image2_exposure)/(invf2*image1_exposure) ) ; % add 1/gradient
l1(counter,:)=  ( J - 1 ) ;  


counter=counter + 1;

end


% A_all=[ A_all ; A ]; 
% l_all=[ l_all ; l ];


A1_all=A1;
l1_all=l1;

n=3;
cvx_begin
    variable x(n)
   
    %minimize( norm(A*x - l) ); %try with and without square
    %minimize( power(norm(A*x ),2) )
    %minimize(power(2,norm(A*x,2)))
    minimize( (A1_all*x -l1_all)'*(A1_all*x - l1_all) )
% with constraint 
    %subject to 
    %sum(x) == 1;
cvx_end

%evector_min=x;
[Vhomo,D]=eig(A1_all'*A1_all);
%% pesudo method 
V = inv(A1_all'*A1_all)*A1_all'*l1_all;

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
%figure('Name','vignette truth vs estimate')
% function vignette truth
eps = 0.005;
radius_map_norm_sub = radius_map_f1(center_I1(1):size(image1,1), center_I1(2):size(image1,2));
vignette_truth_norm_sub = vignette_truth_norm(center_I1(1):size(image1,1), center_I1(2):size(image1,2));

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