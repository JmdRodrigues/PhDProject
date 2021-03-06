from markdown2 import markdown as mkd
from markdown2 import markdown_path as mkd_p
# from bs4 import BeautifulSoup as bs
import matplotlib.pyplot as plt
import weasyprint
import re
import os
from definitions import CONFIG_PATH

class Report:
    def __init__(self, html_template, report_name):

        self.html_template_dir = CONFIG_PATH + "/PDF_generator/html_templates/"

        self.report_name = report_name

        self.template = self.html_template_dir + html_template + ".html"

        self.fig_counts = 0

        #copy template to output html
        with open(self.template) as f:
            data = f.read()

        self.output_tmp = data

        self.output = open(CONFIG_PATH + "/PDF_generator/output_html/" + self.report_name+'.html', "w")


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

    def add_MKD(self, content, extras=False):
        if(extras):
            print("tables")
            html_content = mkd(content, extras=["wiki-tables"])+"{{next_o}}"
        else:
            html_content = mkd(content)+"{{next_o}}"

        pattern = re.compile(r'{{next_o}}')
        self.output_tmp = pattern.sub(html_content, self.output_tmp)

    def add_graph(self, fig, fig_caption, fig_name):
        self.fig_counts+=1

        fig.savefig(CONFIG_PATH+"/PDF_generator/output_html/images/"+fig_name+".png")

        html_content = """
        
        <figure class="fig">
            <img src="../output_html/images/"""+fig_name+""".png">
            <figcaption>""" "Figure" + str(self.fig_counts) + " - " + fig_caption + """</figcaption>
        </figure>
        {{next_o}}
        """
        pattern = re.compile(r'{{next_o}}')
        self.output_tmp = pattern.sub(html_content, self.output_tmp)

    def gen_pdf(self):
        self.output.write(self.output_tmp)
        self.output.close()


        html = weasyprint.HTML(CONFIG_PATH + "/PDF_generator/output_html/" + self.report_name+'.html')
        # html = weasyprint.HTML(string=self.output_tmp)
        CSS = [CONFIG_PATH+"/PDF_generator/CSS/"+css_i for css_i in os.listdir(CONFIG_PATH + "/PDF_generator/CSS")]
        pdf = html.write_pdf(CONFIG_PATH + "/PDF_generator/output_pdf/" + self.report_name + ".pdf", stylesheets=CSS)
