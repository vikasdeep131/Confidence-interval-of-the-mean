
sum_our_2 =0;
 sum_our_3 =0;
sum_bet_2 = 0;
sum_bet_3 = 0;
sum_hoeff_2 = 0;
 sum_hoeff_3 = 0;
sum_eb_mp_2 = 0;
sum_eb_mp_3 = 0;
   function kl = KL_distance(p,q)
       % FOR BERNOULLI USE THE COMMENT ONE BELOW
       kl  = (p)*(log(p/q)) + (1-p)*(log((1-p)/(1-q))) ;
       % p and q are mean.
       % kl = log(q/p) + (p/q) -1; % for exponential
       %kl = (p-q)^2;
       %kl= kl/2;
    end
countter_2  = 0;
countter_3 = 0;
for loop = 1:100

loop
 
    
    p = 0.9;
    sigma = sqrt(p*(1-p));
    delta = .01;
    r(1,1) = binornd(1,p); % it generates exponential random vairable with mean mu.
    phat(1,1) = r(1,1);
    %first sample is done:
    n =1;
    N = 3000;
    n_values = [2000,3000];
    for i = 2:N
            r(i,1) = binornd(1,p);
            n=n+1;
            phat(i,1) = ((n-1)*phat(i-1,1) + r(i,1))/n;
    end
    
    Var = zeros(N,1);
    for n = 1:N
        for i = 1:n
    
                Var(n,1) = Var(n,1) + (r(i,1)- phat(n,1))^2;
            
        end
            Var(n,1) = Var(n,1)/(n-1);
    end
      
    width_MP_EB =zeros(N,1);
    width_hoeffding =zeros(N,1);
    width_bernstien = zeros(N,1);
    
    for n = n_values
      width_MP_EB(n,1) = 2* sqrt((Var(n,1)*2* log(4/delta))/(n));
      width_MP_EB(n,1) = width_MP_EB(n,1) + (14* log(4/delta))/(3*(n-1));
      width_bernstien(n,1) = 2*(sigma * sqrt((2*log(2/delta))/(n)) + (2*log(2/delta))/(3*n));
      width_hoeffding(n,1) = 2*(sqrt((log(2/delta))/(2*n)));
      
    end
    
    
    
    % betting and prpi-eb
    
    V = zeros(N,1);
    for t = 1:N
        for i = 1:t
            if (i == 1)
                V(t,1) = V(t,1) + (r(1,1))^2;
                
            else
                V(t,1) = V(t,1) + (r(i,1)- phat(i-1,1))^2;
            end
        end
            V(t,1) = V(t,1) + .25;
            V(t,1) = V(t,1)/t;
    end
      
    lambda = zeros(N,N);
    %i = n;
    %j = t;
    for n = 1:N
        for t = 2:N
            lambda(n,t) =  2* log(2/delta);
            lambda(n,t) =  lambda(n,t)/n;
            lambda(n,t) = lambda(n,t)/(V(t-1,1));
            lambda(n,t) = sqrt( lambda(n,t));
        end
    end
    
    
    
    
    %for n = n_values
     % width_PrPI-EB(n,1) =  2*(()/())
    %end
    
    
    
    
      
    numbers = linspace(0.00001, 0.99999, 7000);
    count = length(numbers);
    % Loop through the numbers and display them
    W_p = ones(N,count);
    W_n = ones(N,count);
    W   = ones(N,count);
    CI_min = ones(N,1);
    CI_max = zeros(N,1);
    for n = n_values
        for m = 1:count
            for t = 1:N
                W_p(n,m) = W_p(n,m)* (1+ (lambda(n,t)*(r(t,1)-numbers(1,m))));
                W_n(n,m) = W_n(n,m)* (1- (lambda(n,t)*(r(t,1)-numbers(1,m))));
                W(n,m) = 0.5*W_p(n,m) + 0.5* W_n(n,m);
            end
            
             if (W(n,m) < (1/delta))
                 CI_min(n,1) = min(CI_min(n,1), numbers(1,m));
                 CI_max(n,1) = max(CI_max(n,1), numbers(1,m));
                 
                    
              
                 
             end
                 
          
        end
        
       width_betting(n,1) =     CI_max(n,1)-  CI_min(n,1)  + (2/count);
                  
    
    end
    
    % our method
    myfun = @(phat,x,c)  KL_distance(phat,x) -c;
    %cc= log(1/delta);
    %thsold = (cc/2) + 4*log(1+ cc/2 + sqrt(cc) ); %%% for spef for multiple arms and can be improved for single arm result
    %cof_1 = 3;
    %cof_2 = 1;
    %beta = cof_1* (log(cof_2+ log(n))) + 2* (thsold);
    
    for n = n_values
          c_our = (log(5/delta))/n; %%%%
          c_our_CS = (1+ log((2*(1+n))/(delta)))/n;
          fun = @(x) myfun(phat(n,1),x,c_our);
           x0 = [phat(n,1), 0.9999]; % initial interval
           upper_our(n,1)  = fzero(fun,x0);
           fun = @(x) myfun(phat(n,1),x,c_our);
           x0 = [.000001, phat(n,1)]; % initial interval
           lower_our(n,1)  = fzero(fun,x0);
           width_our(n,1) = upper_our(n,1)- lower_our(n,1);
    
            if (p < lower_our(n,1) || p > upper_our(n,1))
                if (n == 2000)
                    countter_2 =countter_2 + 1;
                end
                if (n == 3000)
                    countter_3 =countter_3 + 1;
                end
            end
          
           
           fun = @(x) myfun(phat(n,1),x,c_our_CS);
          x0 = [phat(n,1), 0.9999]; % initial interval
          upper_our_CS(n,1)  = fzero(fun,x0);
           fun = @(x) myfun(phat(n,1),x,c_our_CS);
           x0 = [.000001, phat(n,1)]; % initial interval
           lower_our_CS(n,1)  = fzero(fun,x0);
           width_our_CS(n,1) = upper_our_CS(n,1)- lower_our_CS(n,1);
    
           
    end
    
    % Define the range of sample sizes for plotting
    
    
    % Create a new figure window
    %figure;
    
    % Plot each method with distinct line styles and colors for clarity
    %plot(n_values, width_MP_EB(n_values), 'r-', 'LineWidth', 1); hold on;
    %plot(n_values, width_hoeffding(n_values), 'b--', 'LineWidth', 1);
    %plot(n_values, width_bernstien(n_values), 'g-.', 'LineWidth', 1);
    %plot(n_values, width_betting(n_values), 'k:', 'LineWidth', 1);
    %plot(n_values, width_our(n_values), 'm-', 'LineWidth', 1);
    %plot(n_values, width_our_CS(n_values), 'y-', 'LineWidth', 1);
    % Label the axes and add a title
    %xlabel('n (number of samples)');
    %ylabel('Width');
    %title('Comparison of Width Methods');
    
    % Add a legend to differentiate the methods
    %legend('MP-EB', 'Hoeffding', 'Bernstein', 'Betting', 'Our Method', 'Our CS' , 'Location', 'Best');
    
    % Enable grid for better visualization
   % grid on;
    
    
    
     sum_our_2 = sum_our_2  + width_our(2000,1);
     sum_our_3 = sum_our_3 + width_our(3000,1);
    
     sum_hoeff_2 =  sum_hoeff_2 + width_hoeffding(2000,1);
     sum_hoeff_3 =  sum_hoeff_3 + width_hoeffding(3000,1);
    
    
     sum_eb_mp_2 =  sum_eb_mp_2  + width_MP_EB(2000,1);
     sum_eb_mp_3 =  sum_eb_mp_3  + width_MP_EB(3000,1);
    
     sum_bet_2 = sum_bet_2 + width_betting(2000,1);
     sum_bet_3 = sum_bet_3 + width_betting(3000,1);
    
    
    

end


sum_our_2/100
sum_our_3/100
sum_bet_2/100
sum_bet_3/100
sum_eb_mp_2/100

sum_eb_mp_3/100


sum_hoeff_2/100
sum_hoeff_3/100