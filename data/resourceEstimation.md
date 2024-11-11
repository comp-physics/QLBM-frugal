# Timings
Using SamplerV2 from Qiskit IBM Runtime package
Using IBM Brisbane backend
Optimization level of general preset manager = 3

## Two Circuit LBM
Stream circuit (no bounds): 73.54269496363062 quantum seconds

Stream circuit (with bounds): 73.54269496363062 ?? 

Vorticity circuit (no bounds): 76.20170891770299 quantum seconds

Vorticity circuit (with bounds): 61.560942397004005 ??

Transpiled stream circuit depth on IBM Brisbane: 277828
Transpiled vorticity circuit depth on IBM Brisbane: 165598

Transpiled stream circuit ops on IBM Brisbane: OrderedDict({'rz': 244254, 'sx': 150294, 'ecr': 75401, 'x': 9546, 'barrier': 2})
Transpiled vorticity circuit ops on IBM Brisbane: OrderedDict({'rz': 174196, 'sx': 104894, 'ecr': 49964, 'x': 7732, 'barrier': 2})

Transpiled stream circuit physical runtime = 49593.96 micro seconds
Transpiled vort circuit physical runtime = 28423.60 micro seconds

## One Circuit LBM
Circuit (no bounds): 61.560942397004005 quantum seconds

Transpiled QLBM circuit depth on IBM Brisbane: 1352242
Transpiled QLBM circuit ops on IBM Brisbane: OrderedDict({'rz': 1388645, 'sx': 887232, 'ecr': 428510, 'x': 42619, 'barrier': 6})
Transpiled QLBM circuit physical runtime = 227162.70 micro seconds
