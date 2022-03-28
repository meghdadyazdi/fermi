$(document).ready(function () {
    var chart;
    var getData = $.get('/data');
    getData.done(function (data) {
        globalThis.rawData = data
        console.log('-----data----', data)

        var ctx = document.getElementById('fitting');

        chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.xdata,
                datasets: [{
                    type: 'bubble',
                    label: 'Data2',
                    data: makeBubbles().labels,
                    backgroundColor: "rgba(76,78,80, .7)",
                    borderColor: "transparent"
                }]
            },
            options: {
                plugins: {
                    title: {
                        text: 'Gaussian fit',
                        display: true
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Emission Energy (eV)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Intensity (arb. unit)'
                        }
                    },
                    xAxes: [{
                        type: 'linear',
                        position: 'bottom',
                        ticks: {
                            autoSkip: true,
                            max: Math.max(...makeLabels().array)
                        }
                    }]
                }
            }
        });

        function makeLabels() {
            let arr = data.xdata;
            arr = arr.sort((a, b) => a - b);
            let newarr = arr.map(item => ({ x: item, y: item }));

            return {
                labels: newarr,
                array: arr
            };
        };

        function makeBubbles() {
            let arr = data.ydata;
            let lineLabels = makeLabels().array
            let labels = arr.map((item, i) => ({ x: lineLabels[i], y: item }))
            return { labels, arr };
        };
    })


    function updateChart() {

        var getDataFit = $.get('/data_fit');
        getDataFit.done(function (data) {
            globalThis.fitData = data

            chart.destroy();

            var ctx = document.getElementById('fitting');

            chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.xfit,
                    datasets: [{
                        type: 'line',
                        label: 'Fit',
                        data: data.yfit,
                        fill: false,
                        backgroundColor: "rgba(218,83,79, .7)",
                        borderColor: "rgba(218,83,79, .7)",
                        pointRadius: 0
                    }, {
                        type: 'bubble',
                        label: 'Data',
                        data: makeBubbles().labels,
                        backgroundColor: "rgba(76,78,80, .7)",
                        borderColor: "transparent"
                    }]
                },
                options: {
                    plugins: {
                        title: {
                            text: 'Gaussian fit',
                            display: true
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Emission Energy (eV)'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Intensity (arb. unit)'
                            }
                        },
                        xAxes: [{
                            type: 'linear',
                            position: 'bottom',
                            ticks: {
                                autoSkip: true,
                                max: Math.max(...makeLabels().array)
                            }
                        }]
                    }
                }
            });

            const ydiff = [];


            for (i = 1; i < rawData.ydata.length; i++) {
                ydiff[i] = rawData.ydata[i] - data.yfit[i];
            }

            const diffchart = new Chart(
                document.getElementById('diffchart'), {
                type: 'line',
                data:
                {
                    labels: data.xfit,
                    datasets: [{
                        label: 'Diff',
                        backgroundColor: 'gray',
                        borderColor: 'gray',
                        data: ydiff,
                        
                options: {
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Emission Energy (eV)'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Intensity (arb. unit)'
                            }
                        }
                    }
                }
                    }]
                }
            }
            );
            function makeLabels() {
                let arr = data.xfit;
                // arr = arr.sort((a, b) => a - b);
                let newarr = arr.map(item => ({ x: item, y: item }));

                return {
                    labels: newarr,
                    array: arr
                };
            };

            function makeBubbles() {
                let arr = rawData.ydata;
                let lineLabels = makeLabels().array
                let labels = arr.map((item, i) => ({ x: lineLabels[i], y: item }))
                return { labels, arr };
            };
        })
    }

    $("#fit").click(function () {
        var getDataFit = $.get('/data_fit');
        getDataFit.done(function (data) {
            updateChart();
        });
    });

    $("#fiting").click(function () {
        var getDataFit = $.get('/data_fit');
        getDataFit.done(function (data) {
            updateChart();
        });
    });

});


