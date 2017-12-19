function error=weaklearner(X)
	totLen=length(X);
	ithEle=1;
	subLen=length(find(X==ithEle));
	while(subLen~=0)
		logVal(ithEle)=-(subLen/totLen)*log2(subLen/totLen);
		ithEle=ithEle+1;
		subLen=length(find(X==ithEle));
	end
error=sum(logVal);