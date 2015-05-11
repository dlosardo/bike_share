# First get the working dir and assign to root_dir
root_dir <- getwd()
#' This function sets the working directory to the global variable root_dir
#' Allows for scripts to be run on different machines
cdroot <- function() {
  setwd(root_dir)
}
