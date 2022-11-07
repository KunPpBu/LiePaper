
# First create a TCM using skip grams, we'll use a 5-word window
# most options available on CreateDtm are also available for CreateTcm
tcm <- CreateTcm(doc_vec = MU3D_Video_Level_Data$Transcription,
                 skipgram_window = 10,
                 verbose = FALSE,
                 cpus = 2)

# use LDA to get embeddings into probability space
# This will take considerably longer as the TCM matrix has many more rows 
# than a DTM
embeddings <- FitLdaModel(dtm = tcm,
                          k = 50,
                          iterations = 200,
                          burnin = 180,
                          alpha = 0.1,
                          beta = 0.05,
                          optimize_alpha = TRUE,
                          calc_likelihood = FALSE,
                          calc_coherence = FALSE,
                          calc_r2 = FALSE,
                          cpus = 2)
