function plotzeta(N,M,deltaX,deltaY,x0,zeta)
x = linspace(x0,x0+deltaX*(N-1),N);
y = linspace(0,deltaY*(M-1),M);
z=zeros(length(y),length(x));for i=1:length(x);
for j=1:length(y);z(j,i)=zeta((i-1)*length(y)+j);end;end;
%figure;surf(x,y,z);
yy=zeros(2*length(y),1);
for i=1:length(y);yy(i)=-y(length(y)+1-i);end;for i=length(y)+1:length(yy);
yy(i)=y(i-length(y));end;
zz=zeros(length(yy),length(x));for i=1:length(x);
for j=1:length(y);zz(j,i)=z(length(y)+1-j,i);end;end;
for i=1:length(x);for j=length(y)+1:length(yy);
zz(j,i)=z(j-length(y),i);end;end;
figure;surf(x,yy,zz);%,'facealpha',0.6,'edgecolor','none');
%length(x)
%length(zz(1,:))
%figure;plot(x,zz(end/2,:))

%Dx = @(u)[(u(1:end-4,:)-8*u(2:end-3,:)+0*u(3:end-2,:)-8*u(4:end-1,:)-u(5:end,:))/(12*deltaX)];
%zetax = Dx((zz));
%colormap('gray');
% grid off;

% 
% y  = [0 255];  % luminance
% cb = 255*[1 1]; % chrominance-blue
% cr = 0*[1 1];  % chrominance-red
% map = [linspace(y(1),  y(2),  64)
%        linspace(cb(1), cb(2), 64)
%        linspace(cr(1), cr(2), 64)]' / 255; % in ycbcr space
% rgbmap = ycbcr2rgb(map); % in rgb space
% 
% colormap(rgbmap);
% caxis([-0.11 0.085]);

xlabel('x','FontName','Times New Roman','FontSize',15);
ylabel('y','FontName','Times New Roman','FontSize',15);
zlabel('\zeta','FontName','Times New Roman','FontSize',15,'Rotation',0);
set(gca,'FontName','Times New Roman');
set(gca,'FontSize',15);
ea = min(1,70/(N-1));
c=get(gca,'Children');
set(c,'EdgeAlpha',ea);
%set(gca,'visible','off');
% shading flat;