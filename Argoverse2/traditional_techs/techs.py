from scipy import polyval, polyfit
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d


order = 2

SUPPORTED_TECHS = [
	"poly-fit" ,
	"interp1d", 
	"CubicSpline"
	]
	
def poly_interp(x, y, x_test, order=2):

  params = polyfit(x, y, order)
  y_pred = polyval([*params], x_test)
  return y_pred
  

def interp1d_ex(x, y, x_test):
  func = interp1d(x, y, kind='linear', fill_value='extrapolate')
  y_pred = func(x_test)
  return y_pred


def cubicspline_ex(x, y, x_test, bc_type='not-a-knot'):
  func = CubicSpline(x, y, bc_type=bc_type)
  y_pred = func(x_test)
  return y_pred