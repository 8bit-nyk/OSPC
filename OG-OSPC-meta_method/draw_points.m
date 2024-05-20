        
function draw_points(Ia,Ib,points_img_one,points_img_final,h_lines_num,v_lines_num)
      
        figure ; %clf ;
        imagesc(cat(2, Ia, Ib)) ;

        xa = points_img_one(1,:);
        xb =  points_img_final(1,:) + size(Ia,2) ;
        ya = points_img_one(2,:) ;
        yb = points_img_final(2,:);

        hold on ;
        h = line([xa ; xb], [ya ; yb]) ;
        set(h,'linewidth', 0.5, 'color', 'g') ;
        
        hold on
        xa1 = points_img_one(1,:);
        xb1 =  points_img_final(1,:) ;
        ya1 = points_img_one(2,:) ;
        yb1 = points_img_final(2,:);

        hold on ;
        h = line([xa1 ; xb1], [ya1 ; yb1]) ;
        set(h,'linewidth', 0.05, 'color', 'b') ;
        
        %% draw grid
        hold on
        length=size(Ia,1);
        width=size(Ia,2);
        h_edges_start_y=0:(size(Ia,1)/h_lines_num):size(Ia,1);
        h_edges_start_x=zeros(1,size( h_edges_start_y,1));
        
        
        h_edges_end_y=h_edges_start_y;
        h_edges_end_x=ones(1,size( h_edges_start_y,1))*size(Ia,2);
        
        j = line([h_edges_start_x; h_edges_end_x], [h_edges_start_y ; h_edges_end_y]) ;
        set(j,'linewidth', 1, 'color', 'r') ;
        
        hold on
        
        v_edges_start_x=0:(size(Ia,2)/v_lines_num):size(Ia,2);
        v_edges_start_y=zeros(1,size( v_edges_start_x,1));
        
        
        v_edges_end_y=ones(1,size( v_edges_start_y,1))*length;
        v_edges_end_x=v_edges_start_x;
        
        j = line([v_edges_start_x; v_edges_end_x], [v_edges_start_y ; v_edges_end_y]) ;
        set(j,'linewidth', 1, 'color', 'r') ;
        
        
        
        %v_edges=0:(size(Ia,2)/v_lines_num):size(Ia,2)
        
        
        