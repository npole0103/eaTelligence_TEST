function renderCityRankList(data, containerSelector) {
  const rankList = data;
  console.log(rankList);
  const $lists = $("#" + containerSelector + " .rank-box__list");
  console.log($lists);
  $lists.each(function (i) {
    const $ul = $(this);
    $ul.empty();

    for (let j = 0; j < 5; j++) {
      const item = rankList[i * 5 + j];
      if (!item) continue;

      const $li = $(`
        <li class="rank-box__item">
          <div class="rank-box__item--section">
            <span class="rank-box__item--primary">${item.rank}위</span>
            <span class="rank-box__item--label">${item.region}</span>
          </div>
          <div class="rank-box__item--section">
            <span class="rank-box__item--value">${item.store_count}개</span>
          </div>
        </li>
      `);

      $ul.append($li);
    }
  });
}
