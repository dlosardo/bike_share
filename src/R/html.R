static.dir <- function() {
  'static'
}

asset.path <- function(...) {
  file.path(static.dir(), ...)
}

css <- function(filename) {
  paste("<link rel='stylesheet' type='text/css' href='", asset.path('css', filename) ,"'>", sep='')
}

js <- function(filename) {
  paste("<script type='text/javascript' src='", asset.path('js', filename) ,"'></script>", sep='')
}
js_web <- function(url){
  paste("<script type='text/javascript' src='", url ,"'></script>", sep='')
}
#' Creates an html table with varying arguments
#' @param dat The dataframe
#' @param include.rownames Should row names be included in the table?
#' @param caption A string to be used as the caption
#' @param print.results Should the results be printed or just saved out to an object?
#' @param caption.placement Where to place the caption
htmltable <- function(dat, include.rownames=TRUE, caption = NULL, print.results = TRUE, caption.placement = "bottom") {
  print(xtable(dat
               , caption = caption
               , digits = 2)
        , type="html"
        , include.rownames =include.rownames
        , print.results = print.results
        , caption.placement = caption.placement);
}
