from markdown2 import markdown as mkd
# from bs4 import BeautifulSoup as bs
import matplotlib.pyplot as plt
import weasyprint
import re
import os

class Report:
    def __init__(self, html_template, report_name):

        self.html_template_dir = "../PDF_generator/html_templates/"

        self.report_name = report_name

        self.template = self.html_template_dir + html_template + ".html"

        #copy template to output html
        with open(self.template) as f:
            data = f.read()

        self.output_tmp = data

        self.output = open("../PDF_generator/output_html/" + self.report_name+'.html', "w")


    def add_title(self, title):
        div_title = """

        <h1 class="content_title">"""+title+"""</h1>

        {{next_o}}
        """

        pattern = re.compile(r'{{next_o}}')
        self.output_tmp = pattern.sub(div_title, self.output_tmp)

    def add_subtitle(self, subtitle):
        div_title = """
                <div class="t_div">
                    <h3 class="content_title">""" + subtitle + """</h3>
                </div>
                {{next_o}}
                """

        pattern = re.compile(r'{{next_o}}')
        self.output_tmp = pattern.sub(div_title, self.output_tmp)

    def add_MKD(self, content):
        html_content = mkd(content)+"{{next_o}}"

        pattern = re.compile(r'{{next_o}}')
        self.output_tmp = pattern.sub(html_content, self.output_tmp)

    def add_graph(self, fig, fig_name):
        fig.savefig("../PDF_generator/output_html/images/"+fig_name+".png")

        html_content = """
        <div class="fig">
            <img src="../output_html/images/"""+fig_name+""".png">
        </div>
        {{next_o}}
        """
        pattern = re.compile(r'{{next_o}}')
        self.output_tmp = pattern.sub(html_content, self.output_tmp)

    def gen_pdf(self):
        self.output.write(self.output_tmp)
        self.output.close()


        html = weasyprint.HTML("../PDF_generator/output_html/" + self.report_name+'.html')
        # html = weasyprint.HTML(string=self.output_tmp)
        CSS = ["../PDF_generator/CSS/"+css_i for css_i in os.listdir("../PDF_generator/CSS")]
        pdf = html.write_pdf("../PDF_generator/output_pdf/" + self.report_name + ".pdf", stylesheets=CSS)
