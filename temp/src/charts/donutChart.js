function initDonutChart(elementId, used, unused) {
  const chart = echarts.init(document.getElementById(elementId), null, {
    renderer: "svg",
  });

  const total = used + unused;
  const data = [
    { name: "OKPOS 포스 사용", value: used, itemStyle: { color: "#B0B8C1" } },
    {
      name: "OKPOS 포스 미사용",
      value: unused,
      itemStyle: { color: "#3082F7" },
    },
  ];

  chart.setOption({
    animation: false,
    tooltip: { show: false },
    legend: {
      orient: "vertical",
      right: 110,
      icon: "circle",
      top: "center",
      selectedMode: false,
      itemWidth: 6,
      itemHeight: 6,
      textStyle: {
        fontSize: 11,
        fontWeight: 500,
        lineHeight: 16.5,
        color(value) {
          return value === data[0].name ? "#4E5968" : "#3082F7";
        },
      },
      formatter: (name) => {
        const val = name === "OKPOS 포스 사용" ? used : unused;
        const pct = ((val / total) * 100).toFixed(1);
        return `${name}\n${pct}% (${val}개)`;
      },

      data: data.map((d) => d.name),
    },
    series: [
      {
        type: "pie",
        radius: ["20%", "60%"],
        center: ["35%", "50%"],
        startAngle: 90,
        silent: true,
        legendHoverLink: false,
        label: { show: false },
        labelLine: { show: false },
        data,
      },
    ],
  });

  return chart;
}

$(function () {
  initDonutChart("posSaleTargetChart", 114, 32);
});
