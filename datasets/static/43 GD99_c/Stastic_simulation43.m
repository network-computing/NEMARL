% Create some data
% 定义度数直方图

degree_hist1 = [0, 40, 20, 37, 10, 3, 1];
degree_hist2 = [0, 44, 30, 28, 4, 4, 1];

% 将度数直方图转换为浮点型数组

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
legend('Real data', 'Model simulation')
% title('Wikibooks (Initial)', 'FontSize', 13.2, 'FontWeight', 'bold')
title('GD99\_c', 'FontSize', 13.2, 'FontWeight', 'bold')
% axis([0.03 1.02 0 0.35])
%set(gca,'xtick',[xmin:step:xmax])
% set(gca,'xtick',[1:1000:11000])
set(gca,'xtick', 1:2:max(length(degree_prob1),length(degree_prob2)))
% set(gca,'xticklabels',{'0','0.16','0.32','0.48','0.64','0.80'});
%set(gca,'ytick',[0:0.05:0.35])
xlim([1 max(length(degree_prob1),length(degree_prob2))])
%set(gca,'xscale','log')