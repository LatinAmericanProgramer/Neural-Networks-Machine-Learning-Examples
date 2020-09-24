%% Radial Basis Functions
% Name: Kevin Machado Gamboa
% Course: Neurocontrol 1 
% Professor: David Fernando Ramirez M.
clc;clear all;close all;
%% Note
% This code allow you to 
% 1. The input space dimension is defined by the number of rows
% Tested with 50 patters and 4 centroids ok
% -------------------------------------------------------------------------
%% Initial Parameters and Variables
gaussF = @ (r,sigma) exp((-r.^2)/(2*sigma.^2));     % Gaussian Function
%% Input Space Generation
nP=input('Enter a number of pattern = ');
display('Generation of input samples ...')
x1=((1)+((4)-(1)).*rand(1,nP));
y1=((1)+((4)-(1)).*rand(1,nP));
t1=ones(1,nP);
x2=((-1)+((-4)-(-1)).*rand(1,nP));
y2=((1)+((4)-(1)).*rand(1,nP));
t2=ones(1,nP);t2=t2*2;
x3=((1)+((4)-(1)).*rand(1,nP));
y3=((-1)+((-4)-(-1)).*rand(1,nP));
t3=ones(1,nP);t3=t3*3;
x4=((-1)+((-4)-(-1)).*rand(1,nP));
y4=((-1)+((-4)-(-1)).*rand(1,nP));
t4=ones(1,nP);t4=t4*4;
group1=[x1;y1;t1]; group2=[x2;y2;t2];group3=[x3;y3;t3]; group4=[x4;y4;t4];
X=[group1,group2,group3,group4];
% -------------------------------
figure(1); 
title('Input Space')
hold on
plot(x1,y1,'co',x2,y2,'co',x3,y3,'co',x4,y4,'co');
%% K-means algorithm for Hidden layers
% Input space treatment
[fX,cX]=size(X);                                            % Made to get Dimention
Xx=X(:,randperm(cX));                                       % Random permutation
% Centroid Generation method k-means
T=Xx(fX,:);                                                 % getting Targets
Xx(fX,:)=[];                                                % removing targets from the input space
fX=fX-1;                                                    % getting the correct input space dimension
% 1. Assign the first K input patterns
noCentroid=input('Define the initial # of centroids = ');   % Number of centrois in hidden layer
centroids= Xx(:,randperm(noCentroid));                      % Centroids randomly assigned from inputs values
% 2. Patter distribution: Each pattern belongs to the centroid whose distance is minor
%it=input('Enter a number of iteration = ');                  % Number of iterations
bk=0;
count1=0;
while (bk==0)
newCentroids=zeros(fX,noCentroid);                           % New Centroids storage
distances=zeros(cX,noCentroid);                              % Save distances of each patter and the centroid
countC=zeros(cX,noCentroid);                                 % Count the pattern that belongs to the centroid
centroidSum=zeros(fX,noCentroid);                            % Sum the pattern that belongs to each centroid
for i=1:cX
    for j=1:noCentroid
        distances(i,j)=sum((Xx(:,i)-centroids(:,j)).^2);
    end
    [f,c]=min(distances(i,:));                              % Get the position of the centroid belonging to patter i
    countC(i,c)=1;                                          % Count the pattern that belongs to the centroid
    centroidSum(:,c)=Xx(:,i)+centroidSum(:,c);              % Sum the pattern that belongs to each centroid
end
clear i j;
% 3. New centroids calculus by the average of patterns belonging to each centroid
acumP=sum(countC);                                          % Get the # of pattern per centroid
for i=1:noCentroid
    newCentroids(:,i)=centroidSum(:,i)./acumP(i);           % Get the new centroid
end
clear i;
% 4. Determine if new centroid values have changed with respect to previous centroid values

if abs(newCentroids-centroids)<0.00001
    bk=1;
    break
end
count1=count1+1;
centroids=newCentroids;
end
display(['Number of iterations achieved = ',num2str(count1)])
% -------------------------------
figure(2); 
title('Centroids Position after Training')
hold on
grid on
plot(x1,y1,'co',x2,y2,'co',x3,y3,'co',x4,y4,'co');
plot(newCentroids(1,:),newCentroids(2,:),'*k');
%% Sigma Local Calculus of Hidden Neurons
%display('Starting Centroids Sigma calculation ...')
aS=zeros(noCentroid,max(acumP));            % worst case scenario
sigma=zeros(1,noCentroid);                  % Initialization of sigmas to each centroid
for nc=1:noCentroid
   patron=find(countC(:,nc));
   for i=1:acumP(nc)
       aS(nc,i)=sum((centroids(:,nc)-Xx(:,patron(i))).^2);     
   end
   sigma(1,nc)=sum(aS(nc,:))/acumP(nc);
end
%% Centroid Pruning
% comparing each cetroid with the others
ps=zeros(1,noCentroid);
for i=1:noCentroid
    for j=1:noCentroid
    ps(i,j)= sum((centroids(:,i)-centroids(:,j)).^2);   
    end
end
clear i j;
pru=ps<0.6;
pru=unique(pru,'rows');                         % Returns the same data as in A by rows, but with no repetitions
fCentroids=size(pru);fCentroids=fCentroids(1);  % The final number of centroids
represetative=zeros(fCentroids,fX);             % Initialize vector to storage the represetative centroids
for i=1:fCentroids
    lo=find(pru(i,:)==1);
    for k=1:length(lo)
        represetative(i,:)=represetative(i,:)+newCentroids(:,lo(k))';            
    end
    represetative(i,:)=represetative(i,:)/length(lo);
end
display(['Number of categories found = ',num2str(fCentroids)])
represetative
% -------------------------------
figure(3); 
title('Centroids Position Vs Representative Centroids')
hold on
plot(x1,y1,'co',x2,y2,'co',x3,y3,'co',x4,y4,'co');
plot(newCentroids(1,:),newCentroids(2,:),'*k');
plot(represetative(:,1),represetative(:,2),'r+')
