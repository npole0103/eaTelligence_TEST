function initHorizontalBarChart({ elementId, data }) {
  const chart = echarts.init(document.getElementById(elementId), null, {
    renderer: "canvas",
  });

  const viewData = getDisplayRows(data);

  const yAxisData = viewData.map((d) => `${d.rank}위 ${d.name}`).reverse();
  const seriesData = viewData
    .map((d) => ({
      value: d.value,
      itemStyle: { color: d.isMe ? "#4285F4" : "#BFDBFE" },
    }))
    .reverse();

  chart.setOption({
    animation: false,
    grid: { left: 145, right: 0, top: 0, bottom: 0, containLabel: false },
    xAxis: {
      type: "value",
      splitLine: { show: false },
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { show: false },
    },

    yAxis: {
      type: "category",
      data: yAxisData,
      axisLine: { show: false },
      axisTick: { show: false },
      splitLine: { show: false },
      axisLabel: {
        width: 134,
        overflow: "truncate",
        ellipsis: "…",
        rich: {
          rank: {
            width: 40,
            align: "left",
            fontSize: 12,
            fontWeight: 500,
            color: "#191F28",
          },
          name: {
            align: "left",
            fontSize: 12,
            fontWeight: 600,
            color: "#191F28",
          },
          rankMe: {
            width: 40,
            align: "left",
            fontSize: 12,
            fontWeight: 600,
            color: "#3082F7",
          },
          nameMe: {
            align: "left",
            fontSize: 12,
            fontWeight: 500,
            color: "#3082F7",
          },
        },
        formatter: (value, idx) => {
          const [rank, ...rest] = value.split(" ");
          const name = rest.join(" ");
          const isMe = data[data.length - 1 - idx].isMe;
          const r = isMe ? "rankMe" : "rank";
          const n = isMe ? "nameMe" : "name";
          return `{${r}|${rank}}{${n}|${name}}`;
        },
      },
    },

    series: [
      {
        type: "bar",
        barWidth: 10,
        data: seriesData,
        label: {
          show: true,
          position: "right",
          formatter: "{c}개",
          color: "#4E5968",
          fontSize: 12,
        },
        emphasis: { disabled: true },
      },
    ],
  });
  const bandH = chart.getModel().getComponent("yAxis", 0).axis.getBandWidth();
  const graphics = [];
  for (let i = 1; i <= viewData.length; i++) {
    const centerY = chart.convertToPixel({ yAxisIndex: 0 }, i - 1);
    const y = centerY + bandH / 2 - 0.5;
    graphics.push({
      type: "line",
      shape: { x1: 0, y1: y, x2: chart.getWidth(), y2: y },
      style: { stroke: "#E6E8EA", lineWidth: 1 },
      silent: true,
    });
  }
  chart.setOption({ graphic: graphics }, false);

  return chart;
}
$(function () {
  initHorizontalBarChart({
    elementId: "storeRankChart",
    data: [
      { rank: 1, name: "이삭토스트", value: 411 },
      { rank: 2, name: "에그슬럿", value: 212 },
      { rank: 3, name: "에그슬럿", value: 212 },
      { rank: 4, name: "일상토스트", value: 63 },
      { rank: 5, name: "에그탑", value: 50 },
      { rank: 6, name: "에그탑", value: 50 },
      { rank: 7, name: "에그탑", value: 50 },
      {
        rank: 8,
        name: "에그셀런트",
        value: 114,
        isMe: true,
      },
    ],
  });

  initHorizontalBarChart({
    elementId: "storeSaleRankChart",
    data: [
      { rank: 1, name: "이삭토스트", value: 411 },
      { rank: 2, name: "에그슬럿", value: 212 },
      { rank: 3, name: "에그슬럿", value: 212 },
      { rank: 4, name: "일상토스트", value: 63 },
      {
        rank: 5,
        name: "에그셀런트브랜드이름두줄테스트",
        value: 114,
        isMe: true,
      },
    ],
  });
});
