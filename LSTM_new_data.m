function [phi,Y,V]=LSTM_new_data()      
%生成数据，单入单出系统
%         u(1) = -2+(2+2)*rand(1);
        u(1) = 0;
        y(1) = 0;
        y(2) = 0;
        for k = 2:1001
%             u(k) = -2+(2+2)*rand(1);
%             y(k+1) = 1.5*y(k)*y(k-1)/(1+y(k)*y(k)+y(k-1)*y(k-1))+0.35*sin(y(k)+y(k-1))+1.2*u(k);
%             u(k) = sin(2*pi*k/15)+sin(2*pi*k/25);
            u(k) = sin(pi*k/15)+sin(pi*k/25);
% %             m(k) = 0.6*y(k)-0.2*y(k-1)+1*u(k)+0.5*u(k-1)+1*sin(u(k)+u(k-1)+y(k)+y(k-1))-(u(k)+u(k-1)+y(k)+y(k-1))/(1+u(k)^2+u(k-1)^2+y(k)^2+y(k-1)^2);
% %             v(k) = 1*sin(u(k)+u(k-1)+y(k)+y(k-1))-(u(k)+u(k-1)+y(k)+y(k-1))/(1+u(k)^2+u(k-1)^2+y(k)^2+y(k-1)^2);
%             
            y(k+1) = 1*y(k)-0.5*y(k-1)+2*u(k)+0.7*u(k-1)+2*sin(u(k)+u(k-1)+y(k)+y(k-1))-(u(k)+u(k-1)+y(k)+y(k-1))/(1+u(k)^2+u(k-1)^2+y(k)^2+y(k-1)^2);
            v(k) = 2*sin(u(k)+u(k-1)+y(k)+y(k-1))-(u(k)+u(k-1)+y(k)+y(k-1))/(1+u(k)^2+u(k-1)^2+y(k)^2+y(k-1)^2);
    
%             y(k+1) = m(k)-v(k);
%             y(k+1) = m(k);
            U(:,k-1) = [y(k);y(k-1);u(k);u(k-1)];
%             y(k+1) = m(k)-v(k);
            Y(:,k-1) = y(k+1);
            V(:,k-1) = v(k);
        end
        phi = U;
%         output = Y;

% u(1) = 0;
% y(1) = 0;
% y(2) = 0;
% for k = 2:1000
%     u(k) = sin(2*pi*k/15)+sin(2*pi*k/25);
%     y(k+1) = 1*y(k)-0.5*y(k-1)+2*u(k)+0.7*u(k-1)+2*sin(u(k)+u(k-1)+y(k)+y(k-1))-(u(k)+u(k-1)+y(k)+y(k-1))/(1+u(k)^2+u(k-1)^2+y(k)^2+y(k-1)^2);
%     v(k) = 2*sin(u(k)+u(k-1)+y(k)+y(k-1))-(u(k)+u(k-1)+y(k)+y(k-1))/(1+u(k)^2+u(k-1)^2+y(k)^2+y(k-1)^2);
%     U(:,k-1) = [y(k),u(k)];
%     Y(:,k-1) = y(k+1);
%     V(:,k) = v(k);
% end
%     input = U;
%     output = Y;
