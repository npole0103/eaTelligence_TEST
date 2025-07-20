function getValueByPath(obj, path) {
  return path.split('.').reduce((acc, part) => acc?.[part], obj) ?? '';
}

function renderTemplate(template, data) {
  return template.replace(/{{\s*([\w.]+)\s*}}/g, (_, key) => {
    return getValueByPath(data, key);
  });
}

async function main () {
  const res  = await fetch('./brand/report_data.json');
  const data = await res.json();

  /*
  * requestAnimationFrame()을 두 번 연속으로 사용해 “템플릿을 DOM에 붙인 뒤 2개의 프레임이
  * 완전히 그려진 시점”에 차트를 초기화하고, 기존 인스턴스를 dispose()로 제거했기 때문에
  * 컨테이너 크기가 0이거나 중복 초기화로 깨지는 문제를 피할 수 있습니다.
  * */

  const root = document.getElementById('template-root');
  root.innerHTML = renderTemplate(root.innerHTML, data);   // ① 치환

  /* ② Reflow 가 끝나길 2-frame 동안 기다렸다가… */
  requestAnimationFrame(() => requestAnimationFrame(() => {
    const { brandData, averageData } = data.page1;

    /* ③ 중복 초기화 방지 */
    const dom  = document.getElementById('brandCompetitivenessRadar');
    const prev = echarts.getInstanceByDom(dom);
    if (prev) echarts.dispose(dom); // 기존 인스턴스 제거

    initRadarChart({ elementId: 'brandCompetitivenessRadar', brandData, averageData });
  }));
}
main();
