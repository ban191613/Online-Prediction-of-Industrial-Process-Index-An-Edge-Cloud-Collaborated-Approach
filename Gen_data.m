function [DataSet] =  Gen_data(num)      
% Generate data for numerical simulation experiments
% phi: data vector
% Y: output
% V: unmodeled dynamics
% num: experiment number

u(1) = 0;
y(1) = 0;
y(2) = 0;
for k = 2:1001
    if num == 1   % Exp.1
        u(k) = sin(2*pi*k/15)+sin(2*pi*k/25);
        y(k+1) = 1*y(k)-0.5*y(k-1)+2*u(k)+0.7*u(k-1)+2*sin(u(k)+u(k-1)+y(k)+y(k-1))-(u(k)+u(k-1)+y(k)+y(k-1))/(1+u(k)^2+u(k-1)^2+y(k)^2+y(k-1)^2);
        v(k) = 2*sin(u(k)+u(k-1)+y(k)+y(k-1))-(u(k)+u(k-1)+y(k)+y(k-1))/(1+u(k)^2+u(k-1)^2+y(k)^2+y(k-1)^2);
        
    else if num == 2    % Exp.2
            u(k) = sin(pi*k/15)+sin(pi*k/25);
            y(k+1) = 1*y(k)-0.5*y(k-1)+2*u(k)+0.7*u(k-1)+2*sin(u(k)+u(k-1)+y(k)+y(k-1))-(u(k)+u(k-1)+y(k)+y(k-1))/(1+u(k)^2+u(k-1)^2+y(k)^2+y(k-1)^2);
            v(k) = 2*sin(u(k)+u(k-1)+y(k)+y(k-1))-(u(k)+u(k-1)+y(k)+y(k-1))/(1+u(k)^2+u(k-1)^2+y(k)^2+y(k-1)^2);
        else
            
            u(k) = sin(2*pi*k/15)+sin(2*pi*k/25);   % Exp.3
            y(k+1) = 1*y(k)-0.5*y(k-1)+2*u(k)+0.7*u(k-1)+2*sin(u(k)+u(k-1)+y(k)+y(k-1))-(u(k)+u(k-1)+y(k)+y(k-1))/(1+u(k)^2+u(k-1)^2+y(k)^2+y(k-1)^2)+normrnd(0,0.5);
            v(k) = 2*sin(u(k)+u(k-1)+y(k)+y(k-1))-(u(k)+u(k-1)+y(k)+y(k-1))/(1+u(k)^2+u(k-1)^2+y(k)^2+y(k-1)^2);
        end
    end
    U(:,k-1) = [y(k);y(k-1);u(k);u(k-1)];
    Y(:,k-1) = y(k+1);
    V(:,k-1) = v(k);
end
DataSet  = Y;
end