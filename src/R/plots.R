panel.smooth <- function (x, y) {
  points(x, y)
  lines(stats::lowess(y~x), col="orange", lwd=2.5)
}
## put (absolute) correlations on the upper panels,
## with size proportional to the correlations.
panel.cor <- function(x, y, digits = 2, prefix = "", cex.cor, ...)
{
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  r <- abs(cor(x, y))
  txt <- format(c(r, 0.123456789), digits = digits)[1]
  txt <- paste0(prefix, txt)
  if(missing(cex.cor)) cex.cor <- 0.6/strwidth(txt)
  text(0.5, 0.5, txt, cex = cex.cor * r + .5)
}

plot_empirical_distributions <- function(dat, variables_to_plot){
  dat_melted <- melt(dat, measure.vars=variables_to_plot)
  ggplot(data=dat_melted, aes(x=value, y=..density..)) + 
    geom_histogram(fill="lightblue", color="blue") +
    geom_density() +
    facet_wrap(~variable, scales="free") +
    ggtitle("Empirical Distributions")
}