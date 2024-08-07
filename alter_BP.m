%% Alternate Identification
% L+BP :lambda=1;delta=1;
%% Alternate Identification with Smoothing Factor 
% λ-BP:lambda=0.2;delta=0.2;

clear all
close all

% parameter initialization
Error = 100;    
lambda = 1;     % smoothing factor
delta = 1;      % smoothing factor
num = 1;    % Exp.1
% num = 2;    % Exp.2
% num = 3;    % Exp.3
[input,output,Ve]= Gen_data(num); 
phi = input(:,1:800);
y = output;
v = Ve(:,1:800);
%%
net1=newff(minmax(phi),[4,1],{'tansig','purelin'},'trainlm');  
net1.trainParam.showWindow= false;
net1.trainParam.showCommandLine = false; 
net1.trainParam.show=10;
net1.trainParam.lr=0.05;
net1.trainParam.mc=0.9;
net1.trainParam.goal=0.000001;
net1.trainParam.epochs=100;

%%
theta = [0.8,-0.15,1.8,0.8]';
alpha = 0.1;
Theta = [];
V = [];
be = 300;
for i = 1:be
    % Step2 Edge
    v = (y(i)-phi(:,i)'*theta)*delta;
    if i>1
        v = [Ve(1:i-1),v];
    end
    % Step3 Cloud
    [net1,tr] = train(net1,phi(:,1:i),v);
    % Step4 Cloud
    v_hat = sim(net1,phi(:,i+1));
    V = [V;v_hat];
    % Step5 Edge
    Error = y(i+1)-phi(:,i+1)'*theta-v_hat;
    if abs(Error) > 0.0001
        theta_h = theta+phi(:,i+1)*Error/(alpha+phi(:,i+1)'*phi(:,i+1));
    else
        theta_h = theta;
    end
    % Step6 Edge
    theta = lambda*theta_h+(1-lambda)*theta;
    Theta = [Theta,theta];
end
    
E = [];
for k = 1:be
    y_hat(k) = phi(:,k+1)'*Theta(:,k)+V(k);
    er = y(k+1)-y_hat(k);
    E = [E;er];
end

% Root Mean Squared Error
RMSE = sqrt(E'*E/length(E))
y_re = y(:,2:be+1);
MAPE = sum(abs(E./y_re'))/length(E)

% Root Mean Squared Value
V_re = Ve(:,2:be+1)';
E2 = V_re-V;
RMSV = sqrt(E2'*E2/length(E2));

% Parameter Mean Root Mean Square Value
E3 = [];
theta_hat = mean(Theta,2);
for k = 1:be-1
    e = theta_hat-Theta(:,k);
    E3 = [E3,e];
end
E3 = E3';
for i = 1:4
    esr4(i) = E3(:,i)'*E3(:,i)/length(E3(:,i));
    resr4(i) = sqrt(esr4(i));
end
MRMSV = sum(resr4)


