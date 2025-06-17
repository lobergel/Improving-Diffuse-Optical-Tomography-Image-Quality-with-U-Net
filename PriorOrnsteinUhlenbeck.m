function L1 = PriorOrnsteinUhlenbeck(mesh,std_,r1)

%% Constructs the covariance matrix of prior described in Lieberman et al. 2010
% A. Lipponen 11.9.2012

g1 = mesh.g;
N1 = size(g1,1);

C1 = zeros(N1,N1); % the covariance matrix

for i=1:N1
    Rsq = sqrt((g1(:,1) - g1(i,1)).^2 + (g1(:,2) - g1(i,2)).^2); 
    C1(i,:) = std_^2*exp(-Rsq/r1);   
end


%% Draw samples from the prior

iC1 = inv(C1);
L1 = chol(iC1);
