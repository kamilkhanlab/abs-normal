x = linspace(-30, 30, 100); 
y = x + abs(2*abs(x-2) - 10) + 20;
hy = x + 2*abs(x);

figure
hold on
box on
plot(x, y, 'k')
plot(x, hy, 'k--')
xlabel('x')
ylabel('y')
legend('f(x)','f^{\infty}(\xi)')
hold off

% x = linspace(-30, 30, 100); 
% y = 3*x + abs(2*abs(x-2) - 10) + 20;
% hy = 3*x + 2*abs(x);
% 
% figure
% hold on
% box on
% plot(x, y, 'k')
% plot(x, hy, 'k--')
% xlabel('x')
% ylabel('y')
% legend('f(x)','f^{\infty}(\xi)')
% hold off