function fetchGeoJSON() {
  return koreaGeo;
}

function initGeoChart({ elementId, regionData }) {
  const geoData = fetchGeoJSON();
  const processedGeoJson = replaceNameWithCode(geoData);
  const chartDom = document.getElementById(elementId);
  const chart = echarts.init(chartDom);

  echarts.registerMap("korea", processedGeoJson);

  const option = {
    visualMap: {
      show: false,
      min: 0,
      max: 500,
      inRange: {
        color: ["#e0f3f8", "#66c2a5", "#2c7fb8"],
      },
    },
    geo: {
      map: "korea",
      roam: false,
      silent: true,
      layoutCenter: ["50%", "45%"],
      layoutSize: "100%",
      itemStyle: {
        borderWidth: 0,
        borderColor: "transparent",
      },
    },
    tooltip: { show: false },
    series: [
      {
        type: "map",
        map: "korea",
        geoIndex: 0,
        selectedMode: false,
        label: { show: false },
        emphasis: {
          disabled: true,
          itemStyle: {
            borderWidth: 0,
            borderColor: "transparent",
          },
        },
        itemStyle: {
          borderWidth: 0,
          borderColor: "transparent",
        },
        data: regionData,
      },
    ],
  };

  chart.setOption(option);
  return chart;
}

$(function () {
  initGeoChart({
    elementId: "storeGeoChart",
    regionData: [
      { name: "11110", value: 120 },
      { name: "22010", value: 550 },
      { name: "21310", value: 90 },
      { name: "23030", value: 180 },
      { name: "23040", value: 75 },
      { name: "23320", value: 130 },
      { name: "31080", value: 145 },
      { name: "31230", value: 88 },
      { name: "31380", value: 180 },
      { name: "31270", value: 74 },
      { name: "31110", value: 95 },
      { name: "31200", value: 105 },
      { name: "36110", value: 60 },
      { name: "11230", value: 155 },
    ],
  });
});
