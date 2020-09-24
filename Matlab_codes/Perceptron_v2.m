%% 
% Name: Kevin Machado Gamboa
% Course: Neurocontrol 1
% By: David Fernando Ramirez M.
clc;clear all;close all;
%% Note
%
% -------------------------------------------------------------------------
%% Initial Parameters and Variables
dataSize=1000;
archW=6;
w1=10;
k=10;
r1=10;
d=1;

%% Generating double-moon 
display('Generation input samples ...')
x1=((-13)+(13-(-13)).*rand(1,dataSize/2));
x2=((-3)+(23-(-3)).*rand(1,dataSize/2));
for i=1:dataSize/2
e=((0)+(6-(0)).*rand(1,1));
y1(1,i)=real((sqrt(((e*0.5)+w1)^2-x1(1,i).^2)));
e=((0)+(6-(0)).*rand(1,1));
y2(1,i)=real(-(d+sqrt(((e*0.5)+r1)^2 -(x2(1,i)-k).^2)));
end
figure(1)
hold on;grid on
plot(x1,y1,'*',x2,y2,'+')
title('The double-moon classification problem: Training Examples')
% Shufling Data
input1=[x1;y1]'; input2=[x2;y2]'; X=[input1;input2];
t1=ones(dataSize/2,1); t2=-1*ones(dataSize/2,1); t=[t1',t2'];
space=[X,t'];space=space(randperm(dataSize),:);
X=space(:,[1:2]);t=space(:,3)';X(:,3)=ones(dataSize,1);
% ----------------------------
clear input1 input2 t1 t2
% -------------------------------------------------------------------------
display('Starting Training ... ')
%[w,y]=trainPerceptron(X,t,1,0.001); % Function made to validate Perceptron
[fi,ci]=size(X);
w=rand(1,ci);                               % Sinaptic weights
threshold=0;
funcH=@(y) (-1)*(y<threshold)+(1)*(y>=threshold);            % Activation funcion Heaviside
y=rand(1,fi);                                % Initializing output vector
yy=funcH(y);                                 % Initializing output vector

maxEp=1000;                                   % Maximum number of epochs
vErr=[];                                	 % Error Units Vectors

acum=0;
acum1=0;
i=0;
epochs=0;
lr=0.5;
lro=0.01;                                    % Initial Learning rate
lrf=0.0010;                                % Final Learning rate
Error=zeros(fi,1);                          % Inizializing Error Vector
expecError=0.0001;

xb=[-15:0.1:25];
yb=-xb*(w(1)/w(2))-(w(3)/w(2));
plot(xb,yb)
% -------------------------------------------------------------------------
while acum<=fi
   acum1=acum1+1;
   i=i+1;
   y(1,i)=sum(w.*X(i,:));          % Partial output
   yy(1,i)=funcH(y(1,i));          % Global output y
   delta=(t(1,i)-yy(1,i));         % Indicates the needs of an actualization
   Error(i,1)=delta^2;
            ww=w+(lr*delta*X(i,:));% Learning rate rule PseudoInverse
            w=ww;
   if i==fi                          % Devuelve la cuenta       
      pE=sum(Error(:,1))/fi;
      if pE<expecError
            	acum=fi+5;         	% Debe ser mayor a cuatro para terminar
                display('Network Achieve Expected error')
      end
      i=0;
      epochs=epochs+1;                       % Increase the number of epochs
      lr=lro+(lrf-lro)*(epochs/maxEp);       % Linear Variation of Learning Rate
      vErr(epochs)=pE;        % Vector of quadratic errors
   end  
   % Break the while if it is too much
   if epochs==maxEp;
       display('Network output doesnt match the expected error')
       break                                           
   end
end
% -------------------------------------------------------------------------
% Get the accuracy of training phase
display(['Number of Epochs: ' num2str(epochs)])
v1=(t-yy);
V2=find(v1==0);
nnHits=size(V2);
pHits=100*(nnHits(2)/fi);
display([num2str(nnHits(2)) ' Hits from ' num2str(fi)]);
display(['Neural Network Training Accuracy = ' num2str(pHits), '%']); 
% -------------------------------------------------------------------------
figure
plot(vErr);title('Error Evolution');xlabel('Epochs');ylabel('Error');
%% Validation Phase of Neural Network
dataSize=1000;
display('...')
display('...')
display('Generating Data For Validation ...')
x1v=((-13)+(13-(-13)).*rand(1,dataSize/2));
x2v=((-3)+(23-(-3)).*rand(1,dataSize/2));
for i=1:dataSize/2
e=((0)+(6-(0)).*rand(1,1));
y1v(1,i)=real((sqrt(((e*0.5)+w1)^2-x1v(1,i).^2)));
e=((0)+(6-(0)).*rand(1,1));
y2v(1,i)=real(-(d+sqrt(((e*0.5)+r1)^2 -(x2v(1,i)-k).^2)));
end

figure(3)
hold on;grid on
plot(x1v,y1v,'*',x2v,y2v,'+')
title('The double-moon classification problem: Validation Examples')
% Shufling Data
input1v=[x1v;y1v]'; input2v=[x2v;y2v]'; Xv=[input1v;input2v];
t1v=ones(dataSize/2,1); t2v=-1*ones(dataSize/2,1); tv=[t1v',t2v'];
spacev=[Xv,tv'];spacev=spacev(randperm(dataSize),:);
Xv=spacev(:,[1:2]);tv=spacev(:,3)';Xv(:,3)=ones(dataSize,1);
clear input1v input2v t1v t2v
% -------------------------------------------------------------------------
display('Staring Validation ...')
% ------------------------------------------------------------------
funcH=@(y) (-1)*(y<0)+(1)*(y>0);            % Activation funcion Heaviside
y=rand(1,fi);                               % Initializing output vector
yy=funcH(y);                                % Initializing output vector

    for p=1:fi
        y(1,p)=sum(w.*Xv(p,:));           % Partial output
        yy(1,p)=funcH(y(1,p));            % Global output y 
    end
    
v1=(tv-yy);
V2=find(v1==0);
nnHits=size(V2);
pHits=100*(nnHits(2)/fi);
display([num2str(nnHits(2)) ' Hits from ' num2str(fi)]);
display(['Neural Network Validation Accuracy = ' num2str(pHits), '%']);    
 
% ------------------------------------------------------------------
%% decision Boundary plot
figure(3)
hold on
xb=[-15:0.1:25];
yb=-xb*(w(1)/w(2))-(w(3)/w(2));
plot(xb,yb)