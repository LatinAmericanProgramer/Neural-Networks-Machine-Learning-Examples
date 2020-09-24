%% Perceptron
% Name: Kevin Machado Gamboa
% Course: Neurocontrol 1
% By: David Fernando Ramirez M.
clc;clear all;close all;hold off;
% -------------------------------------------------------------------------
%% Code Description 
% The next code shows a perceptron with two input neurons and one output
% neuron using as training algorithm the LMS (Least Mean Square)
% The Reference can be found in page 57, Equation 2.28 
% Redes Neuronales y Sistemas Borrosos 3ra Edicion. Bonifacio Martin del Brio
% NOTE: this code DOESN'T have a Weight Matrix
clc;clear all;close all;
display('To start working with this code, first think in a Vector Target (vT) ')
display('vT its a row Vector composed by 4 elements with "-1" & "1" as possibles values ')
% -------------------------------------------------------------------------
X=[0,0,1;0,1,1;1,0,1;1,1,1];               	 % Input X
t=[0,0,0,0];                                 % Target
t(1)=input('enter the 1st element of target E(-1 o 1) = ');
t(2)=input('enter the 2nd element of target E(-1 o 1) = ');
t(3)=input('enter the 3th element of target E(-1 o 1) = ');
t(4)=input('enter the 4th element of target E(-1 o 1) = ');
[fi,ci]=size(X);                            % Input size to get the sinaptic weights
w=rand(1,ci);                               % Sinaptic weights
lr=0.1;%input('Learning Rate Value = ');         % Learning rate
maxEp=10000;%input('Maximun number of epochs allowed');
ErrorRate=0.1;%input('Error Rate To Achieve= '); % Error Rate

funcH=@(y) (-1)*(y<0)+(1)*(y>=0);           % Activation funcion Heaviside
y=rand(1,fi);                               % Initializing output vector
yy=funcH(y);                                % Initializing output vector
vecError=[];                                % Inizializing Error Vector
Error=zeros(fi,1);                          % Inizializing Error Vector

p=0;
acum=0;
acum1=0;
i=0;
epocas=0;
display('Output before training phase')
yy
display('Target')
t
while acum<=4
   acum1=acum1+1;
   i=i+1;
   y(1,i)=sum(w.*X(i,:));          % Partial output
   yy(1,i)=funcH(y(1,i));          % Global output y
   delta=(t(1,i)-yy(1,i));         % Indicates the needs of an actualization
   Error(i,1)=delta;
        if delta~=0
            p1=lr*X(i,:);p2=t(1,i)-w.*X(i,:);  % Arguments p1 and p2 LMS
            ww=w+(p1).*(p2);w=ww;              % LMS rule

            acum=0;
        else
            acum=acum+1;            % Acumula el numero aciertos seguidos 
        end     
   if i==4                          % Devuelve la cuenta
      sumErr=sum(Error)/fi;
      p=p+1;
      vecError(1,p)=sumErr;
       i=0;
      
   end
   epocas=epocas+1;                 % Increase the number of epochs
   % Break the while if it is too much
   if acum1==maxEp;
       display('Network output doesnt match the target')
       break                                           
   end
end
%% Plots
% next changes -1 by 0 to plot the decision boundary
tt=t;
for i=1:fi
    if t(1,i)==-1
       tt(1,i)=0; 
    end
end
plotpv(X(:,[1:2])',tt);
plotpc(w,1);
hold on
xb=[-15:0.1:25];
yb=-xb*(w(1)/w(2))-(w(3)/w(2));
plot(xb,yb,'r')
display('Epochs ')
epocas
display('output after training')
yy