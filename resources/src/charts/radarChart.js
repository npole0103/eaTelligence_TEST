function initRadarChart({ elementId, brandData, averageData }) {
  const chartDom = document.getElementById(elementId);
  const chart = echarts.init(chartDom);

  const option = {
    animation: false,
    tooltip: { show: false },
    legend: {
      orient: "vertical",
      right: 0,
      top: 0,
      selectedMode: false,
      textStyle: {
        color: "#4e5968",
        fontSize: 11,
        fontWeight: 500,
      },
      itemWidth: 18,
      itemHeight: 4,
      data: [
        {
          name: "대상 브랜드",
          icon: "rect",
          itemStyle: {
            color: "#a24bdc",
          },
        },
        {
          name: "업계 평균",
          icon: "rect",
          itemStyle: {
            color: "#6b7684",
          },
        },
      ],
    },
    radar: {
      indicator: [
        { name: "수익성", max: 100 },
        { name: "안정성", max: 100 },
        { name: "잠재성", max: 100 },
        { name: "확장성", max: 100 },
      ],
      radius: 80,
      splitNumber: 4,
      shape: "polygon",
      axisName: {
        color: "#4E5968",
        fontSize: 11,
      },
      splitLine: {
        lineStyle: { color: "#d1d6db" },
      },
      splitArea: {
        areaStyle: { color: ["#fff"] },
      },
      axisLine: {
        lineStyle: { color: "#e5e8eb" },
      },
      silent: true,
    },
    series: [
      {
        type: "radar",
        data: [
          {
            value: brandData,
            name: "대상 브랜드",
            lineStyle: {
              color: "#A234C7",
              width: 2,
            },
            areaStyle: {
              color: "rgba(162, 52, 199, 0.1)",
            },
            symbol: "none",
          },
          {
            value: averageData,
            name: "업계 평균",
            lineStyle: {
              color: "#6B7684",
              type: "dashed",
              width: 1,
            },
            areaStyle: {
              color: "rgba(78, 89, 104, 0.1)",
            },
            symbol: "none",
          },
        ],
      },
    ],
  };

  chart.setOption(option);
  return chart;
}

$(function () {
  initRadarChart({
    elementId: "brandCompetitivenessRadar",
    brandData: [85, 90, 70, 95], // 대상 브랜드의 수치
    averageData: [60, 65, 55, 70], // 업계 평균 수치
  });
});