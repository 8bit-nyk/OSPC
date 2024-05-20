function  region_image=create_region_img(num_blocks_x,num_blocks_y,num_rows,num_cols)

    % num_blocks_x= 3  ;
    % num_blocks_y= 3  ;
    % 
    % num_rows=  200   ;
    % num_cols=  100   ;

    region_image=zeros( num_rows,num_cols );

    edges_cols=ceil(1:(num_cols/num_blocks_x):num_cols);
    edges_cols=[edges_cols num_cols  ];
    edges_rows=ceil(1:(num_rows/num_blocks_y):num_rows);
    edges_rows=[edges_rows num_rows  ];

    list_num_regions=1:(num_blocks_x*num_blocks_y);

    region_number= 1;

    for region_v=1:num_blocks_y 
      for region_h=1:num_blocks_x



        %% fill the region 
        start_row=edges_rows(region_v   );
        end_row=edges_rows( region_v  + 1);

        start_col=edges_cols( region_h  );
        end_col=edges_cols( region_h + 1);

        %% could replace the two for loops 
        for col=start_col:end_col
            for row=start_row:end_row


                region_image(row,col)=region_number;

            end
        end



        %for next region
        region_number= region_number + 1;


      end
    end
