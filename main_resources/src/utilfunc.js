// 지도 json 파일의 name 속성을 지역코드로 변경하는 함수
function replaceNameWithCode(geoJson) {
  geoJson.features.forEach((f) => {
    f.properties.name = f.properties.code;
  });
  return geoJson;
}

// 내 가게 포함 5개만 리턴하는 함수
function getDisplayRows(raw) {
  const list = [...raw].sort((a, b) => a.rank - b.rank);
  const meIdx = list.findIndex((d) => d.isMe);
  if (meIdx === -1 || meIdx < 5) {
    return list.slice(0, 5);
  }
  return [...list.slice(0, 4), list[meIdx]];
}
