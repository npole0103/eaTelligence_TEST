from src.service.metaDataService import metaMain, metaFormatting
from src.service.page1DataService import page1Main
from src.service.page2DataService import page2Main

# BRD_20190835
meta = metaMain('BRD_20190835', 202412)
page1 = page1Main('BRD_20190835', 202412, meta)
page2 = page2Main('BRD_20190835', 202412, meta)

# formatting
meat = metaFormatting(meta)