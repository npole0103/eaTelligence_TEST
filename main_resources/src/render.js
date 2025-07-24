function getValueByPath(obj, path) {
    return path.split(".").reduce((acc, part) => acc?.[part], obj) ?? "";
}

function renderTemplate(template, data) {
    return template.replace(/{{\s*([\w.]+)\s*}}/g, (_, key) => {
        return getValueByPath(data, key);
    });
}

async function main() {
    console.log("main 실행됨");

    const res = await fetch("./brand/report_data.json");
    const data = await res.json();

    const root = document.getElementById("template-root");

    const radarDom = root.querySelector("#brandCompetitivenessRadar");
    const storeGeoDom = root.querySelector("#storeGeoChart");
    const storeCountDom = root.querySelector("#storeCountChart");
    const salesCountDom = root.querySelector("#salesCountChart");
    const storeRankDom = root.querySelector("#storeRankChart");
    const storeSaleRankDom = root.querySelector("#storeSaleRankChart");
    const posSaleTargetDom = root.querySelector("#posSaleTargetChart");

    if (radarDom) {
        const prev = echarts.getInstanceByDom(radarDom);
        if (prev) echarts.dispose(prev);
    }

    if (storeGeoDom) {
        const prev = echarts.getInstanceByDom(storeGeoDom);
        if (prev) echarts.dispose(prev);
    }

    if (storeCountDom) {
        const prev = echarts.getInstanceByDom(storeCountDom);
        if (prev) echarts.dispose(prev);
    }

    if (salesCountDom) {
        const prev = echarts.getInstanceByDom(salesCountDom);
        if (prev) echarts.dispose(prev);
    }

    if (storeRankDom) {
        const prev = echarts.getInstanceByDom(storeRankDom);
        if (prev) echarts.dispose(prev);
    }

    if (storeSaleRankDom) {
        const prev = echarts.getInstanceByDom(storeSaleRankDom);
        if (prev) echarts.dispose(prev);
    }

    if (posSaleTargetDom) {
        const prev = echarts.getInstanceByDom(posSaleTargetDom);
        if (prev) echarts.dispose(prev);
    }

    root.innerHTML = renderTemplate(root.innerHTML, data);

    requestAnimationFrame(() =>
        requestAnimationFrame(() => {
            renderCardColors();

            // 페이지 1
            const {brandData, averageData} = data.page1;
            initRadarChart({
                elementId: "brandCompetitivenessRadar", brandData, averageData,
            });

            // 페이지 2
            const {regionData} = data.page2;
            initGeoChart({elementId: "storeGeoChart", regionData});

            const {xAxisData: xAxisData2, yAxisData: yAxisData2} = data.page2.store_trend;
            initStoreCountChart({elementId: "storeCountChart", xAxisData: xAxisData2,  yAxisData: yAxisData2})
            renderCityRankList(data.page2.store_cnt_rank_by_city , "storeRankList");

            // 페이지 3
            const {xAxisData: xAxisData3, yAxisData: yAxisData3} = data.page3.sales_trend;
            initSalesChart({elementId: "salesCountChart", xAxisData: xAxisData3, yAxisData: yAxisData3})
            renderCityRankList(data.page3.amt_rank_by_city, "saleRankList");

            // 페이지 4
            const {store_data, sales_data} = data.page4;
            console.log(store_data)
            console.log(sales_data)
            initHorizontalBarChart({elementId: "storeRankChart", data: store_data});
            initHorizontalBarChart({elementId: "storeSaleRankChart", data: sales_data});

            // 페이지 5
            const {used, unused} = data.page5;
            initDonutChart("posSaleTargetChart", used, unused);
        })
    );
}

main();
