%% Self Organized Map: Equilateral Triangle Representaton
% By: Kevin Machado Gamboa
% Email: ing.kevin@hotmail.com; kevin.machado@uao.edu.co
% Created: April 26, 2017
% Modify: April 27, 2017
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
% note: tested with: 
% Numero de patrones por lado en el triangulo 40
% Nro total de neuronas=100 
% Nro de veces que se presentan las entradas=500
clc;clear all;close all
lad = @(h) h/cot(60);
lado1 = @ (x,h) (x/cot(60)) + h;
lado2 = @ (x,h) (-x/cot(60)) + h;
% Lado 1 del triangulo
x=input('Patterns per each side of Triangle / Numero de patrones por lado en el triangulo ?');
h=x/cot(60);
la1=[-(x-1):1:0];
a1=lado1(la1,h);
plot(la1,a1,'r*')
hold on
% Lado 2 del triangulo
la2=[0:1:x-1];
a2=lado2(la2,h);
plot(la2,a2,'r*')
hold on
% Base del triangulo
xbase=[-x:2:x];
xbase(xbase==0)=[];                 % Remove the Zero component
ybase=zeros(1,length(xbase));
plot(xbase,ybase,'r*')

nump=x*3;

% -------------------------------------------------------------------------
entrdx=[la1,la2,xbase];
entrdy=[a1,a2,ybase];
redpun=[entrdx;entrdy];
% -------------------------------------------------------------------------
neutot=input('total number of neurons / Nro total de neuronas: neutot=');
neux=x/2;
neuy=sqrt(3*x)/2;
pesosx=neux+[sign(0.5-rand(1,neutot))].*[(neux/10)*rand(1,neutot)];
pesosy=neuy+[sign(0.5-rand(1,neutot))].*[(neuy/10)*rand(1,neutot)];
pesosi=[pesosx;pesosy];

Np=input('Epochs / Nro de veces que se presentan las entradas Np=');
Rxi=0.5*ceil(1*min(neux,neuy));
alphai=0.9;
alphaf=0.1;
Rf=0;
% -------------------------------------------------------------------------

% ----------------------
clear i
redpun=[entrdx;entrdy];
% ----------------------
figure(1)
title('Initial Position of Neurons')
plot(redpun(1,:),redpun(2,:),'r*'); 
hold on
plot(pesosi(1,:),pesosi(2,:),'b*')
% ---------------------
redpunpt=redpun;
pesos=pesosi;

for i=1:Np*neutot;
    indexs=randperm(nump); %Revolver
    redpunpt=redpun(:,indexs);
    k=mod(i,neutot)*(mod(i,neutot)~=0)+(neutot)*(mod(i,neutot)==0);
    for j=1:neutot;
        normaresta(j)=norm(redpunpt(:,k)-pesos(:,j));
    end
    clear j
    Rx(i)=Rxi+(Rf-Rxi)*((i-1)/(Np*neutot));
    alpha(i)=alphai+(alphaf-alphai)*((i-1)/(Np*neutot));
    winner(i)=find(normaresta==min(normaresta),1);
    pesos(:,winner(i))=pesos(:,winner(i))+ alpha(i)*(redpunpt(:,k)-pesos(:,winner(i)));
    for j=1:neutot;
        if abs(winner(i)-j)<=Rx(i);
           pesos(:,j)=pesos(:,j)+ alpha(i)*(redpunpt(:,k)-pesos(:,j));
        end
    end
end

figure(2)
plot(redpun(1,:),redpun(2,:),'r*'); 
title('Position of Neurons Vs Input Space After Training')
hold on
plot(pesos(1,:),pesos(2,:),'b*');
hold off
figure(3)
plot(pesos(1,:),pesos(2,:),'b*');
title('Final Position of Neurons')