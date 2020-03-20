function [err,estLabels] = missRate(trueLabels,estLabels)
    estLabels = myBestMap(trueLabels,estLabels);
    err  = sum(trueLabels ~= estLabels) / length(trueLabels);

function outLabels = myBestMap(trueLabels,estLabels)

trueLabelVals = unique(trueLabels);
kTrue = length(trueLabelVals);
estLabelVals = unique(estLabels);
kEst = length(estLabelVals);

cost_matrix = zeros(kEst,kTrue);
for ii = 1:kEst
    inds = find(estLabels == estLabelVals(ii));
    for jj = 1:kTrue
        cost_matrix(ii,jj) = length(find(trueLabels(inds) == trueLabelVals(jj)));
    end
end

[rInd,cInd] = linear_sum_assignment(-cost_matrix);

outLabels = inf*ones(size(estLabels));
for ii = 1:length(rInd)
    outLabels(estLabels==estLabelVals(rInd(ii))) = trueLabelVals(cInd(ii));
end

outLabelVals = unique(outLabels);
if length(outLabelVals) < max(outLabels)
    lVal = 1;
    for ii = 1:length(outLabelVals)
        outLabels(outLabels==outLabelVals(ii)) = lVal;
        lVal = lVal + 1;
    end
end