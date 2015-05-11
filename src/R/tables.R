#' Creates an overall descriptives table using the `describe` function in the
#'  `psych` package
#' @param dat A dataframe
#' @param vars_to_exclude A String vector of variable names to exclude from table
#' @return An html formatted table of descriptives
overall_descriptives_table <- function(dat, vars_to_exclude){
  tmp_table <- describe(dat[, names(dat)[!names(dat) %in% vars_to_exclude]])[
    , c("n", "mean", "sd", "median", "min", "max", "skew", "kurtosis", "se")]
  tmp_table$n <- as.integer(tmp_table$n)
  htmltable(tmp_table
    , caption="Overall Descriptives", caption.placement="top")
}
#' Very specific function to create a table of descriptives by categorical variables
#' @param dat A dataframe that MUST have the following variables:
#'  `season`, `weather`, `holiday`, `workingday`, `count`, `temp`, `atemp`
#'  `humidity`, `windspeed`, `casual`, `registered`
descriptives_by_categorical_vars_table <- function(dat){
  cat_vars_desc <- ddply(dat, .(season, weather, holiday, workingday), summarize
                     , count_for_order = mean(as.numeric(count))
                     , count = paste0(mean_rounded(count, 2), " (", sd_rounded(count, 2), ")")
                     , temp = paste0(mean_rounded(temp, 2), " (", sd_rounded(temp, 2), ")")
                     , atemp = paste0(mean_rounded(atemp, 2), " (", sd_rounded(atemp, 2), ")")
                     , humidity = paste0(mean_rounded(humidity, 2), " (", sd_rounded(humidity, 2), ")")
                     , windspeed = paste0(mean_rounded(windspeed, 2), " (", sd_rounded(windspeed, 2), ")")
                     , casual = paste0(mean_rounded(casual, 2), " (", sd_rounded(casual, 2), ")")
                     , registered = paste0(mean_rounded(registered, 2), " (", sd_rounded(registered, 2), ")")                   
  )
  cat_vars_desc <- cat_vars_desc[order(cat_vars_desc$count_for_order, na.last=T, decreasing=T), ]
  htmltable(cat_vars_desc[, names(cat_vars_desc)[!names(cat_vars_desc) %in% "count_for_order"]], include.rownames=F
            , caption="Mean and SD of Count and Various Features by Season, Weather, Holiday, and Working Day"
            , caption.placement="top")
}