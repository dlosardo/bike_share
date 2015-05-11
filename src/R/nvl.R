#' If value is either NA or an empty string, return default, else return value
#' @param value An object to be tested against it being NA or an emptry string
#' @param default The default object to return if value is NA or an empty string
nvl <- function(value, default) {
  ifelse(is.na(value) | value == '', default, value);
}
