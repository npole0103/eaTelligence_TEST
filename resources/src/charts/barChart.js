/* ───────────────────────── 매출 추이 차트 ───────────────────────── */
function initSalesChart({
  elementId,
  xAxisData,
  yAxisData,
  baseColor = "#FEEECD",
  prevColor = "#FBB322",
  lastColor = "#d6d6d6",
}) {
  const chart = echarts.init(document.getElementById(elementId));

  const data = yAxisData.map((v) => v / 1e8);
  const last = data.length - 1;
  const prev = last - 1;

  const seriesData = data.map((v, i) => ({
    value: v,
    itemStyle: {
      color: i === last ? lastColor : i === prev ? prevColor : baseColor,
    },
  }));

  chart.setOption({
    animation: false,
    grid: { top: 30, right: 0, bottom: "0%", left: "0%", containLabel: true },

    xAxis: {
      type: "category",
      data: xAxisData,
      axisTick: { show: false },
      axisLine: { lineStyle: { color: "#eee" } },
      axisLabel: { color: "#6B7684", fontSize: 8 },
    },

    yAxis: {
      type: "value",
      name: "매출액(억)",
      nameTextStyle: { color: "#6B7684", fontSize: 8 },
      splitNumber: 6,
      boundaryGap: [0, 0.1],
      axisLine: { show: false },
      axisLabel: {
        color: "#6B7684",
        fontSize: 8,
        formatter: (v) => `${v}억`,
      },
      splitLine: { lineStyle: { color: "#eee" } },
    },

    series: [
      {
        type: "bar",
        barWidth: "10px",
        emphasis: { disabled: true },
        data: seriesData,
      },
    ],
    tooltip: { show: false },
  });

  return chart;
}

/* ───────────────────────── 점포 수 차트 ───────────────────────── */
function initStoreCountChart({
  elementId,
  xAxisData,
  yAxisData,
  baseColor = "#89D8D8",
  lastColor = "#24BFBF",
}) {
  const chart = echarts.init(document.getElementById(elementId));

  const last = yAxisData.length - 1;
  const seriesData = yAxisData.map((v, i) => ({
    value: v,
    itemStyle: { color: i === last ? lastColor : baseColor },
  }));

  chart.setOption({
    animation: false,
    grid: { top: 30, right: 0, bottom: "0%", left: "0%", containLabel: true },

    xAxis: {
      type: "category",
      data: xAxisData,
      axisTick: { show: false },
      axisLine: { lineStyle: { color: "#eee" } },
      axisLabel: { color: "#6B7684", fontSize: 8 },
    },

    yAxis: {
      type: "value",
      name: "점포 수(개)",
      nameTextStyle: { color: "#6B7684", fontSize: 8 },
      splitNumber: 6,
      boundaryGap: [0, 0.1],
      axisLine: { show: false },
      axisLabel: { color: "#6B7684", fontSize: 8 },
      splitLine: { lineStyle: { color: "#eee" } },
    },

    series: [
      {
        type: "bar",
        barWidth: "10px",
        emphasis: { disabled: true },
        data: seriesData,
      },
    ],
    tooltip: { show: false },
  });

  return chart;
}
$(function () {
  initStoreCountChart({
    elementId: "storeCountChart",
    xAxisData: [
      "24.01",
      "24.02",
      "24.03",
      "24.04",
      "24.05",
      "24.06",
      "24.07",
      "24.08",
      "24.09",
      "24.10",
      "24.11",
      "24.12",
    ],
    yAxisData: [140, 12, 97, 96, 95, 94, 93, 93, 92, 91, 91, 92],
  });

  initSalesChart({
    elementId: "salesCountChart",
    xAxisData: [
      "24.01",
      "24.02",
      "24.03",
      "24.04",
      "24.05",
      "24.06",
      "24.07",
      "24.08",
      "24.09",
      "24.10",
      "24.11",
      "24.12",
      "예상",
    ],
    yAxisData: [
      1400000, 8000000, 14000000, 1400000, 1400000, 1400000, 9100003, 1400000,
      1400000, 10000000, 1400000, 1400000, 1400000,
    ],
  });
});
