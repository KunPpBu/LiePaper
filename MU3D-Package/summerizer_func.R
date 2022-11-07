summarizer <- function(doc, gamma) {
  
  # recursive fanciness to handle multiple docs at once
  if (length(doc) > 1 )
    # use a try statement to catch any weirdness that may arise
    return(sapply(doc, function(d) try(summarizer(d, gamma))))
  
  # parse it into sentences
  sent <- stringi::stri_split_boundaries(doc, type = "sentence")[[ 1 ]]
  
  names(sent) <- seq_along(sent) # so we know index and order
  
  # embed the sentences in the model
  e <- CreateDtm(sent, ngram_window = c(1,1), verbose = FALSE, cpus = 2)
  
  # remove any documents with 2 or fewer words
  e <- e[ rowSums(e) > 2 , ]
  
  vocab <- intersect(colnames(e), colnames(gamma))
  
  e <- e / rowSums(e)
  
  e <- e[ , vocab ] %*% t(gamma[ , vocab ])
  
  e <- as.matrix(e)
  
  # get the pairwise distances between each embedded sentence
  e_dist <- CalcHellingerDist(e)
  
  # turn into a similarity matrix
  g <- (1 - e_dist) * 100
  
  # we don't need sentences connected to themselves
  diag(g) <- 0
  
  # turn into a nearest-neighbor graph
  g <- apply(g, 1, function(x){
    x[ x < sort(x, decreasing = TRUE)[ 3 ] ] <- 0
    x
  })
  
  # by taking pointwise max, we'll make the matrix symmetric again
  g <- pmax(g, t(g))
  
  g <- graph.adjacency(g, mode = "undirected", weighted = TRUE)
  
  # calculate eigenvector centrality
  ev <- evcent(g)
  
  # format the result
  result <- sent[ names(ev$vector)[ order(ev$vector, decreasing = TRUE)[ 1:3 ] ] ]
  
  result <- result[ order(as.numeric(names(result))) ]
  
  paste(result, collapse = " ")
}