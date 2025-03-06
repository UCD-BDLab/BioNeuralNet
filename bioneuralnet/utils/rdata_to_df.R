args <- commandArgs(trailingOnly = TRUE)
input_file <- args[1]
csv_file <- args[2]
object_name <- args[3]

orig_input <- input_file
orig_csv <- csv_file

rm(list = setdiff(ls(), c("orig_input", "orig_csv")))

success <- tryCatch({
  load(orig_input)
  loaded_objs <- ls()
  
  if (length(loaded_objs) == 1) {
    mat <- get(loaded_objs[1])
  } else if (exists("AdjacencyMatrix")) {
    mat <- AdjacencyMatrix
  } else if (exists("M")) {
    mat <- M
  } else if(exists(object_name)) {
    mat <- get(object_name)
  }
  else {
    stop("Error: Neither 'Object','AdjacencyMatrix' or 'M' was found in the RData file.")
  }
  
    message("Attempting to write CSV to: ", orig_csv)
    write.csv(mat, file = orig_csv, row.names = TRUE)
    if (file.exists(orig_csv)) {
    message("CSV file successfully written to: ", orig_csv)
    } else {
    message("CSV file was NOT created at: ", orig_csv)
    }

  TRUE
}, error = function(e) {
  message("Skipping network conversion for file: ", orig_input)
  message("Error: ", e$message)
  FALSE
})

quit(save = "no", status = 0)
