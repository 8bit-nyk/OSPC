function [I1]=correct_avg(I1,I2,I3,I4)
%         clc
%         clear all
% clc
% clear all

% I1=imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_30\images\000" + 36 + ".jpg");
% I2=imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_30\images\000" + 37 + ".jpg");
% I3=imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_30\images\000" + 38 + ".jpg");
% I4=imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_30\images\000" + 39 + ".jpg");

%% put the frames in layers
I=zeros(size(I1,1),size(I1,2),4);
y=0;
x=0;

I(:,:,1)=I1;
I(:,:,2)=I2;
I(:,:,3)=I3;
I(:,:,4)=I4;
%% for each frame
total_num_frames=4;
repeated_arr=zeros(size(I(:,:,1)));
f1_f_other_arr_x=zeros( size(I(:,:,1),1) , size(I(:,:,1),2) , 3    );
f1_f_other_arr_y=zeros( size(I(:,:,1),1) , size(I(:,:,1),2) , 3    );

for frame=1:total_num_frames-1 %% start from closest frame till the farthest
    
%%% get matches
[matchedPointsf1_r,matchedPointsf2_r]=get_matches(uint8(I(:,:,frame)),uint8(I(:,:,frame+1)));

%% fill repeated_arr
for element=1:size(matchedPointsf1_r,1)
    [y,x]=deal( double(matchedPointsf1_r(element,2)) , double(matchedPointsf1_r(element,1)) ); 
    if repeated_arr(y,x)< frame && repeated_arr(y,x) > frame - 2
        repeated_arr(y,x)=repeated_arr(y,x) + 1 ;                              %% assume first is inclusive (big assumption,failed) %% sometimes corr is repeated im same pair (weird)
        %% fill f_i_fi+1_arr_x and f_i_fi+1_arr_y 
        f1_f_other_arr_x(y,x,frame)=matchedPointsf2_r(element,1);
        f1_f_other_arr_y(y,x,frame)=matchedPointsf2_r(element,2);
    end
end
%% end loop 
end

%% time to take average 
I1_flatted=I1(:);
I2_flatted=I2(:);
I3_flatted=I3(:);


f12x=f1_f_other_arr_x(:,:,1);
f12x=f12x(:);
f12y=f1_f_other_arr_y(:,:,1);
f12y=f12y(:);

f13x=f1_f_other_arr_x(:,:,2);
f13x=f13x(:);
f13y=f1_f_other_arr_y(:,:,2);
f13y=f13y(:);

f14x=f1_f_other_arr_x(:,:,3);
f14x=f14x(:);
f14y=f1_f_other_arr_y(:,:,3);
f14y=f14y(:);


%% repeated 3 times 
% repeated3_index=find(repeated_arr(:)==3); %% search for find for 2d 
% for element=1:size(repeated3_index,1)
%     index=repeated3_index(element);
%     I1_flatted(index)=  ( I1_flatted(  index ) + I2( f12y(index) ,f12x(index) ) + I3( f13y(index) ,f13x(index) ) + I4( f14y(index) ,f14x(index) ) ) / 4;
% end
% 
% %% repeated 2 times 
% repeated2_index=find(repeated_arr(:)==2); %% search for find for 2d 
% for element=1:size(repeated2_index,1)
%     index=repeated2_index(element);
%     I1_flatted(index)=  ( I1_flatted(  index ) + I2( f12y(index) ,f12x(index) ) + I3( f13y(index) , f13x(index) )  ) / 3;
% end
test_I1=[];
test_I2=[];
%% repeated 1 times 
repeated1_index=find(repeated_arr(:)==1); %% search for find for 2d 
for element=1:size(repeated1_index,1)
    index=repeated1_index(element);
    I1_flatted(index)=  ( I1_flatted(  index ) + I2( f12y(index) ,f12x(index) )  ) / 2;
    %%just for test
    test_I1=[test_I1 I1_flatted(  index )];  %% we should remove outlierssssss
    test_I2=[test_I2 I2_flatted(  index )];
end


I1_restored=reshape(I1_flatted,size(I1));
    
    
    