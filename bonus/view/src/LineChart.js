import { useEffect, useState } from "react";
import { Chart } from "react-google-charts";
import axios from "axios";

const hostname = window.location.hostname;
const port = "8000";
const link_to_webserver = `http://127.0.0.1:8000`
 /// 'http://127.0.0.1:8000';

function LineChart() {
  const data = [
    ["x", "Variable 2", "Variation 3"],
    [0, 0, 0],
    [1, 10, 5],
    [2, 23, 15],
    [3, 17, 9],
    [4, 18, 10],
    [5, 9, 5],
    [6, 11, 3],
    [7, 27, 19],
  ];
  /// X axis is toggeable, this means the data is also toggleable

  const options = {
    hAxis: {
      // x axis
      title: "X axis",
    },
    vAxis: {
      // y axis
      title: "Production of Cereal (POC)",
    },
    series: {
      1: { curveType: "function" },
    },
  };

  const [variationData, setVariationData] = useState([]);
  const [predictorVariable, setPredictorVariable] = useEffect({

  })
  const calculate_ypred = (predictorVar) => {

  }

  const min_max={
    "CONSTANT": [0,10], 
    "FDI": [-3,5],
    "POP": [10,70],
    "TEMP": [0,6]
  }

  useEffect(() => {
    async function fetchData() {
      try {
        const resp = await axios.get(`${link_to_webserver}/data`);
        setVariationData(resp.data);
        console.log(resp.data)
      } catch (err) {
        console.error("Error fetching data from webserver:\n", err);
      }
    }
    fetchData()
  }, []);

  
  return (
    <>
      {link_to_webserver}
      
      <Chart
        chartType="LineChart"
        width="100%"
        height="400px"
        data={data}
        options={options}
      />
    </>
  );
}

export default LineChart;

/*
{
  "variation_2": {
    "max_r2_score_row": {
      "CONSTANT": 4.891033690288461,
      "FDI": 0.0431438772505104,
      "MSE_SCORE": 0.0196485682550691,
      "POP": 0.396796964548904,
      "R2_SCORE": 0.829902145409924,
      "TEMP": -0.0502394011471151
    }
  },
  "variation_3": {
    "max_r2_score_row": {
      "CONSTANT": 4.855973186069275,
      "FDI": 0.0136807502659397,
      "MSE_SCORE": 0.015968505336586,
      "POP": 0.3464331658854209,
      "R2_SCORE": 0.8508456418058515,
      "TEMP": -0.0151577564995059
    }
  }
}
*/