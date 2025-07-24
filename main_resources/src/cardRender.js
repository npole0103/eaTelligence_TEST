function renderCardColors() {
  $(".card__wrap--rank .card")
    .slice(-2)
    .each(function () {
      const $valueEl = $(this).find(".card__item--value").last();
      const level = $valueEl.text().trim();

      if (level === "높음") $valueEl.addClass("fc-blue");
      else if (level === "보통") $valueEl.addClass("fc-gray");
      else if (level === "낮음") $valueEl.addClass("fc-red");
    });
}
