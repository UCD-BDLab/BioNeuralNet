args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
  stop("Usage: Rscript extract_CVfold.R <output_path>")
}

output_path <- args[1]
options(scipen = 999)

success <- tryCatch({
  CVdata_path <- file.path(output_path, "CVFold.Rdata")
  load(CVdata_path)

  network_path <- file.path(output_path, "globalNetwork.Rdata")
  load(network_path)

  export_base <- file.path(output_path, "CV_Export")
  dir.create(export_base, showWarnings = FALSE, recursive = TRUE)

  for (f_name in names(folddata)) {

    fold_dir <- file.path(export_base, f_name)
    dir.create(fold_dir, showWarnings = FALSE)

    current_fold <- folddata[[f_name]]

    for (i in 1:length(current_fold$X_train)) {
      write.csv(current_fold$X_train[[i]],
                file = file.path(fold_dir, paste0("X_train_Omics_", i, ".csv")),
                row.names = FALSE)
    }

    for (i in 1:length(current_fold$X_test)) {
      write.csv(current_fold$X_test[[i]],
                file = file.path(fold_dir, paste0("X_test_Omics_", i, ".csv")),
                row.names = FALSE)
    }

    if (is.list(current_fold$Y_train)) {
      for (i in 1:length(current_fold$Y_train)) {
        write.csv(current_fold$Y_train[[i]],
                  file = file.path(fold_dir, paste0("Y_train_Omics_", i, ".csv")),
                  row.names = FALSE)
      }
    } else {
      write.csv(current_fold$Y_train,
                file = file.path(fold_dir, "Y_train.csv"),
                row.names = FALSE)
    }

    if (is.list(current_fold$Y_test)) {
      for (i in 1:length(current_fold$Y_test)) {
        write.csv(current_fold$Y_test[[i]],
                  file = file.path(fold_dir, paste0("Y_test_Omics_", i, ".csv")),
                  row.names = FALSE)
      }
    } else {
      write.csv(current_fold$Y_test,
                file = file.path(fold_dir, "Y_test.csv"),
                row.names = FALSE)
    }

    message("Finished exporting: ", f_name)
  }

  message("All files (X and Y) saved to: ", export_base)

  write.csv(globalNetwork$AdjacencyMatrix,
            file.path(output_path, "globalNetwork_R_matrix.csv"),
            row.names = TRUE)

  TRUE

}, error = function(e) {
  message("Error during extraction: ", e$message)
  FALSE
})

if (!success) {
  quit(save = "no", status = 1)
} else {
  quit(save = "no", status = 0)
}
