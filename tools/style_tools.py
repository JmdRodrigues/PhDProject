from colorutils import Color, ArithmeticModel
import matplotlib.colors as mcolors

color_list = ["dodgerblue", "orangered", "lightgreen", "mediumorchid", "gold", "firebrick", "darkorange", "springgreen", "lightcoral"]
color_list2 = list(mcolors.CSS4_COLORS)
primary_colors = {"d":Color(web="#e3c44c", arithmetic=ArithmeticModel.BLEND), "a":Color(web="#d73824", arithmetic=ArithmeticModel.BLEND), "c":Color(web="#66CB5E", arithmetic=ArithmeticModel.BLEND), "b":Color(web="#6e91ee",arithmetic=ArithmeticModel.BLEND)}
