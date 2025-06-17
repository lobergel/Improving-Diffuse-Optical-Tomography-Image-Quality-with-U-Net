function PlotFields(g,H,phi,clim);



% figure(figh)

% axes(axh)



nn = size(g,1);

ne = size(H,1);



% hold on

% set(gca,'defaultpatchfacecolor',[0 0 0]);

set(gcf,'defaultpatchedgecolor','none');

%colorbar

G = g(H,:);

G = reshape(G(:),ne,6);

Z = phi(H);

cmin = clim(1);
cmax = clim(2);

caxis([cmin cmax])

patch(G(:,1:3)',G(:,4:6)',Z',Z')

% shading interp

%view(2)

% axis('equal')

axis('off')

colormap('jet')
