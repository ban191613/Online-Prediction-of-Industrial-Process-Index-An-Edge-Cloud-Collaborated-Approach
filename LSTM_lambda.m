% 带平滑因子的交替辨识
clear all
close all

% 初始化参数
Error = 100;
alpha = 0.1;
lambda = 0.98;  %平滑因子
delta = 0.1;
% theta = [0.55,-0.15,1.05,0.45]';
theta = [0.85,-0.4,1.8,0.6]';
train  = [];
Theta = [];
V = [];

% 加载数据
[Phi,output,Ve] = LSTM_new_data();
% 对数据进行标准化处理
% [phi,is] = mapminmax(Phi);
% [y,os] = mapminmax(output);
phi = Phi;
y = output;

% 初始化LSTM
numFeatures = 4;
numResponses = 1;
numHiddenUnits = 30;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
options = trainingOptions('adam', ...
    'MaxEpochs',1000, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.01, ...
    'Verbose',0);

% 进行交替辨识
be = 300;
for k = 1:be
     %Step1:求出v的导师信号
    v = (y(k)-phi(:,k)'*theta)*delta;
    if k>1
            v = [Ve(1:k-1),v];
    end
    %Step2:训练网络求v的估计值
    net1 = trainNetwork(phi(:,1:k),v,layers,options);
    [net1,v_hat]=predictAndUpdateState(net1,phi(:,k+1));
    V = [V;v_hat];
    Error = y(k+1)-phi(:,k+1)'*theta-v_hat;
    %Step3:更新线性部分的参数
    if abs(Error) > 0.0001
            theta_h = theta+phi(:,k+1)*Error/(alpha+phi(:,k+1)'*phi(:,k+1));
    else
            theta_h = theta;
    end
    theta = lambda*theta_h+(1-lambda)*theta;
    Theta = [Theta,theta];
end
% 参数辨识结果
figure(15)
plot(Theta(1,:));
hold on 
plot([1,k],[1,1],'k');
hold on 
plot(Theta(2,:));
hold on
plot([1,k],[-0.5,-0.5],'k');
legend('a1_hat','a1','a2_hat','a2');

figure(16)
plot(Theta(3,:));
hold on 
plot([1,k],[2,2],'k');
hold on 
plot(Theta(4,:));
hold on
plot([1,k],[0.7,0.7],'k');
legend('b1_hat','b1','b2_hat','b2');

% 预测误差
E = [];
for k = 1:be
    y_hat(k) = phi(:,k+1)'*Theta(:,k)+V(k);
    er = y(k+1)-y_hat(k);
    E = [E;er];
end

% 均方根误差
esr = E'*E/length(E);
resr = sqrt(esr)
y_re = y(:,2:be+1);
MAPE = sum(abs(E./y_re'))/length(E)

%鏈未建模动态的估计误差
V_re = Ve(:,2:be+1)';
E2 = V_re-V;
esr2 = E2'*E2/length(E2);
resr2 = sqrt(esr2)
MAPE2 = sum(abs(E2./V_re))/length(E2)


