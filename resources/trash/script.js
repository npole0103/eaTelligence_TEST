fetch("script_data.json")
  .then(response => response.json())
  .then(data => {
      console.log("script_data.json 로드 완료")
      const container = document.getElementById("content");
      data.forEach(item => {
        const block = document.createElement("div");
        block.className = "card";
        block.innerHTML = `
          <h3>${item.title}</h3>
          <p><strong>Name:</strong> ${item.name}</p>
          <p><strong>Age:</strong> ${item.age}</p>
          <p><strong>Description:</strong> ${item.description}</p>
        `;
        container.appendChild(block);
    });
  });


// ✅ 템플릿 치환용 sample_data.json 처리
fetch("html_data.json")
  .then(response => response.json())
  .then(data => {
    console.log("html_json.json 로드 완료")
    const templateEl = document.getElementById("template-example");
    const rawTemplate = templateEl.innerHTML;

    const rendered = renderTemplate(rawTemplate, data);
    templateEl.innerHTML = rendered;
  });

// {{key}} 치환 함수
function renderTemplate(template, data) {
  return template.replace(/{{\s*(\w+)\s*}}/g, (_, key) => data[key] ?? "");
}