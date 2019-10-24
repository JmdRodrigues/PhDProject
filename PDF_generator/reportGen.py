from jinja2 import FileSystemLoader, Environment
import weasyprint
import re
import os

from requests import request

class Report:
    def __init__(self, html_template, report_name):

        self.html_template_dir = "/media/jeanraltique/FishStory/Projectos/Doutoramento/PhDCode/PhDProject/PDF_generator/html_templates/"

        self.report_name = report_name

        self.template = self.html_template_dir + html_template + ".html"

        #copy template to output html
        with open(self.template) as f:
            data = f.read()

        self.output_tmp = data

        self.output = open("/media/jeanraltique/FishStory/Projectos/Doutoramento/PhDCode/PhDProject/PDF_generator/output_reports/"+self.report_name+".html", "w")


    def add_title(self, title):
        div_title = """
        <div class="title">
            <h3 class="masthead-title" style="display: flex;justify-content: left; align-items: center;">
                <a style="display: inline-block; padding-left: 3%">"""+title+"""</a>
            </h3>
        </div>
        {{next_o}}
        """


        pattern = re.compile(r'{{next_o}}')
        self.output_tmp = pattern.sub(div_title, self.output_tmp)

    def gen_pdf(self):
        self.output.write(self.output_tmp)
        self.output.close()


        html = weasyprint.HTML("/media/jeanraltique/FishStory/Projectos/Doutoramento/PhDCode/PhDProject/PDF_generator/output_reports/" + self.report_name+'.html', base_url="/media/jeanraltique/FishStory/Projectos/Doutoramento/PhDCode/PhDProject/PDF_generator/header_images")
        # html = weasyprint.HTML(string=self.output_tmp)
        CSS = ["/media/jeanraltique/FishStory/Projectos/Doutoramento/PhDCode/PhDProject/PDF_generator/CSS/"+css_i for css_i in os.listdir("/media/jeanraltique/FishStory/Projectos/Doutoramento/PhDCode/PhDProject/PDF_generator/CSS")]
        pdf = html.write_pdf(self.report_name + ".pdf", stylesheets=CSS)
