import PDF_generator.reportGen as rg

doc = rg.Report("libphys", "test_report")

doc.add_title("Love this spectacular thing")

doc.gen_pdf()