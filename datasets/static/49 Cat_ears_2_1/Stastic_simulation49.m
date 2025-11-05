% Create some data
% 定义度数直方图

degree_hist1 = [0, 5, 0, 1, 17, 17, 19, 21, 11];
degree_hist2 = [0, 5, 3, 6, 17, 17, 16, 18, 6, 3];

% 将度数直方图转换为浮点型数组
 [0, 105, 92, 79, 51, 47, 24, 14, 6, 5, 2, 3, 4, 2, 1, 1, 1, 1, 2]
degree_hist1 = double(degree_hist1);
degree_hist2 = double(degree_hist2);

degree1 = degree_hist1(2:end);
degree2 = degree_hist2(2:end);
% 计算度数概率
degree_prob1 = degree1 / sum(degree1);
degree_prob2 = degree2 / sum(degree2);

figure

set(gca, 'FontSize', 12);

  
plot(1:length(degree_prob1), degree_prob1, '^-','Color',[65/256,105/256,225/256],'LineWidth',1.2,'Markersize',7,'MarkerFaceColor',[65/256,105/256,225/256])

hold on
%plot(eb, predictionData,'ro-','LineWidth',1.2,'Markersize',7,'MarkerFaceColor','r')
%plot(eb, predictionData, 's-','Color',[171/256,130/256,255/256],'LineWidth',1.2,'Markersize',7,'MarkerFaceColor',[171/256,130/256,255/256]) 
%plot(eb, epsilon7, 's-','Color',[0/256,206/256,209/256],'LineWidth',1.2,'Markersize',7,'MarkerFaceColor',[0/256,206/256,209/256])

%plot(eb, size700, 's-','Color',[171/256,130/256,255/256],'LineWidth',1.2,'Markersize',7,'MarkerFaceColor',[171/256,130/256,255/256]) %Purple
plot(1:length(degree_prob2), degree_prob2, 's-','Color',[255/256,106/256,106/256],'LineWidth',1.2,'Markersize',7,'MarkerFaceColor',[255/256,106/256,106/256])



% Turn on the grid
grid on

% Add title and axis labels
%title('Performance of Baseband QPSK')
xlabel('Node degree k','FontSize',13.2,'FontWeight','bold')
ylabel('Degree probability p(k)','FontSize',13.2,'FontWeight','bold')
% 将图例位置设置
legend('Real data', 'Model simulation', 'Location', 'northwest')
% title('Wikibooks (Initial)', 'FontSize', 13.2, 'FontWeight', 'bold')
title('Cat\_ears\_2\_1', 'FontSize', 13.2, 'FontWeight', 'bold')
% axis([0.03 1.02 0 0.35])
%set(gca,'xtick',[xmin:step:xmax])
% set(gca,'xtick',[1:1000:11000])
set(gca,'xtick', 1:2:max(length(degree_prob1),length(degree_prob2)))
% set(gca,'xticklabels',{'0','0.16','0.32','0.48','0.64','0.80'});
%set(gca,'ytick',[0:0.05:0.35])
xlim([1 max(length(degree_prob1),length(degree_prob2))])
%set(gca,'xscale','log')