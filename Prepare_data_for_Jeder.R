require('reshape')
input_dir <- '/media/mahfuz/2A8B1AF404AA6CF6/CRISPR/GIN_Analysis/GIN_Paper_Analysis/Datasets/qGI_data/'
setwd('/media/mahfuz/2A8B1AF404AA6CF6/CRISPR/MCMC_Jeder/Jeder_Manuscript/Analysis/WT_Minimal_analysis/Input')

# Read minimal and rich screen information (wtScreen_ID)
load(paste0(input_dir, 'wtScreen_ID.rda')) # wtScreen_ID$min # Minimal screens

# Read WT data (wtLFC_perScreen)
load(paste0(input_dir, 'wtLFC_perScreen_2021_05_14.rda'))

# Generate sets of screens (3,5,7,10,15, and 21)
random_seed <- 40
num_of_screens <- c(3,4,5,7,10,15,21)

for (i in seq_along(num_of_screens)){
	set.seed(random_seed)
	ind_of_screen <- sample(length(wtScreen_ID$min), size = num_of_screens[i], replace = F)

	data_WT <- wtLFC_perScreen[,wtScreen_ID$min[ind_of_screen],'mean']
	tmp1 <- as.data.frame(melt(data_WT), stringsAsFactors = FALSE)
	colnames(tmp1) <- c('expid', 'repid', 'lfc')

	# Average number of essential (at lfc < -1.0)
	print(sum(data_WT < -1.0, na.rm = T) / dim(data_WT)[2]) # 2006, 2045, 2085, 2070, 2079, 2085, 2080
	
	# Average number of essential (at lfc < -2.0)
	print(sum(data_WT < -2.0, na.rm = T) / dim(data_WT)[2]) # 751, 774, 827, 817, 833, 841, 835

	write.table(tmp1, paste0('lfc_WT_min_rep_', length(ind_of_screen), '_seed_40.txt'), sep = '\t', row.names = FALSE)
}
