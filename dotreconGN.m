function [x,logpost] = dotreconGN(m,x,geom,hreg,niter)
% minimization of  G(x) = || m - F(x) ||^2 + P(x)
% for DOT
% inputs;
% m; data (model m = F(x) + e, e is noise )  
% x: initial guess
% hreg; struct containing handles to P(x) (prior)
% geom; struct containing the geometry & known parameters
% niter; max number of Gauss-Newton steps
% -----------------------------------------------
% -- GN stopping criteria parameters ---

posttol = 1e-12;
gradtol = 1e-12;
difftol = 1e-12;
emuatol = 1;

% -----------------------------------------------
logpost = zeros(niter+1,1);
norm0 = logp(x,m,geom,hreg);
it = 1;
logpost(it) = norm0;
gnorm = gradtol + 1;   % temporary setting to pass first "while" 
resdiff = difftol + 1; % temporary setting to pass first "while" 

while any([it-1 >= niter,  norm0 < posttol, gnorm < gradtol, resdiff < difftol]) == 0;  
  
  disp(['Iteration ' num2str(it)])  
  it = it + 1;
  disp(' ------------------------------')
  disp('constructing gradient & Hessian of G(x) ...')
  
  [gpost,H] = gradpost(x,m,geom,hreg);  
  
  disp('--------------------------------')  
  gnorm = norm(gpost);      
  disp('solving Gauss-Newton direction ...')
  dx = H\gpost;
  
  disp('----------------------------------------')  
  disp(' entering linesearch  ...')
  
  s = linesearch(dx,norm0,x,m,geom,hreg);

  disp('----------------------------------------')  
  disp(['step length = ' num2str(s)])
  disp('------------------------------------')
  
  x = x + s * dx;
  x = max(x,1e-6);
  norm0 = logp(x,m,geom,hreg);
  logpost(it) = norm0;    
  resdiff = logpost(it-1) - logpost(it);  
  
end
logpost = logpost(1:it);

% ------------------------------
function val = logp(x,m,geom,hreg);

% -- computes value of log posterior  
n = geom.hMesh.NodeCount();
pval1 = 0.5*norm(hreg.Lmua*(x(1:n)-hreg.x(1:n)))^2;
pval2 = 0.5*norm(hreg.Lmus*(x(n+1:end)-hreg.x(n+1:end)))^2;

m0 = ForwardToast(x,geom);  % forward solution
lval = 0.5 * norm(hreg.Le * (m - m0))^2;
val = lval + pval1 + pval2;

function [g,H] = gradpost(x,m,geom,hreg);  

% --- computes gradient of log posterior ---
n = geom.hMesh.NodeCount();
[m0,J] = ForwardToast(x,geom);
J = hreg.Le * J;
dm = hreg.Le * (m - m0);

% --- gradient of  G(x)

gradp1 = (hreg.Lmua'*hreg.Lmua)*((x(1:n))-hreg.x(1:n));
gradp2 = (hreg.Lmus'*hreg.Lmus)*((x(n+1:end))-hreg.x(n+1:end));
g = J'*dm - [gradp1;gradp2];

% ---- Hessian of G(x)  
H = hreg.H;
H = J' * J + H;
clear J

function s = linesearch(dx,norm0,x,m,geom,hreg);
  % syntax for log posterior;  
  a = 0.5;  
  % --------
  s0 = 0; 
  d0 = norm0; 
  flag = 0;
  tol = 1e-6;
  % --- Compute initial (s1,d1)
  s1 = a;
  xtmp = x + s1*dx;
  xtmp = max(xtmp,tol);  
  d1 =  logp(xtmp,m,geom,hreg);     
  % ------ Main body ----------
  if d1 > d0,  
    % --- Set s1 --> s2, half the initial s1 as s1 = s1/2
    s2 = s1; d2 = d1;
    s1 = s1/2;
    xtmp = x + s1 * dx;
    xtmp = max(xtmp,tol);    
    d1 =  logp(xtmp,m,geom,hreg); 
    while d1 > d0 && flag == 0,
      s2 = s1; d2 = d1;
      s1 = s1/2; 
      xtmp = x + s1 * dx;
      xtmp = max(xtmp,tol);
      d1 =  logp(xtmp,m,geom,hreg);        
      if s1 < tol,	
	flag = 1;
	break    
      end
    end
  else,  
    % --- set initial s2 = 2*s1
    s2 = 2 * s1;
    xtmp = x + s2 * dx;
    xtmp = max(xtmp,tol);    
    d2 =  logp(xtmp,m,geom,hreg);     
    while d2 < d1,
      % --- set s0=s1, s1=s2, s2=2*s1;
      s0 = s1; d0 = d1;
      s1 = s2; d1 = d2;
      s2 = 2*s2;
      xtmp = x + s2 * dx;
      xtmp = max(xtmp,tol);      
      d2 =  logp(xtmp,m,geom,hreg); 
    end
  end
  if flag == 0,
    % --- fit curve ax^2+bx+c to the data (s0,s1,s2),(d0,d1,d2)
    svec = [s0;s1;s2];
    hmat = [svec.^2 svec ones(3,1)];
    th = hmat\[d0;d1;d2];    
    % -- step length s = -b/2a
    s = -th(2)/(2*th(1));
    % -- ys needed for plotting only
    ys = th(1)*s^2+th(2)*s+th(3);    
  else,
    s = tol; % tol/10;
    disp(['using s = ' num2str(s) ' (tolerance reached in the search)'])  
  end
  % --------------
  
 function [z,J] = ForwardToast(x,geom)
  n = geom.hMesh.NodeCount();  
  frq = geom.frq;
  muag = x(1:n);
  musg = x(n+1:end);  
  refind = geom.ref;
  K = dotSysmat(geom.hMesh,muag,musg,refind,frq);
  dphi = K\geom.qvec;
  gamma = geom.mvec.' * dphi;
  % convert projections to log
  zz = log(gamma(:));
  % split real and imaginary parts
  lamp = real(zz);
  phs = imag(zz);
  
  z = [lamp;phs];
  if nargout > 1        
    disp(' === calculating Jacobian ==== ')            
    J = toastJacobian(geom.hMesh,0,geom.qvec,geom.mvec,muag,musg,refind,frq);
    sca = 0.3/1.4;
    J(:,1:n) = J(:,1:n) * sca;
    J(:,n+1:end) = J(:,n+1:end) * diag(-sca ./ (3*(muag + musg).^2));
    J(:,1:n) = J(:,1:n) + J(:,n+1:end);
  end
% ==========================
