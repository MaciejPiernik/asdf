import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'

const url = "https://pierniol-asdf.hf.space/predict"
const HF_TOKEN = import.meta.env.VITE_HF_TOKEN

const getHeaders = () => ({
  'Authorization': `Bearer ${HF_TOKEN}`,
  'Content-Type': 'application/json'
})

// ask it to predict please
function askPrediction(file) {
  const formData = new FormData();
  formData.append('file', file);

  return fetch(url, {
    method: "POST",
    headers: getHeaders(),
    body: formData
  })
  .then(response => response.json())
  .then(result => {
    console.log("Prediction result:", result)
    return result
  })
  .catch(error => {
    console.error("Error during prediction:", error)
  })
}

function App() {
  const [file, setFile] = useState(null)
  const [prediction, setPrediction] = useState(null)

  const handleFileChange = (e) => {
    if (e.target.files) {
      setFile(e.target.files[0])
    }
  }

  const handlePredict = async () => {
    if (file) {
      const result = await askPrediction(file)
      if (result) {
        setPrediction(result.predicted_digit)
      }
    }
  }

  return (
    <>
      <div>
        <a href="https://vite.dev" target="_blank">
          <img src={viteLogo} className="logo" alt="Vite logo" />
        </a>
        <a href="https://react.dev" target="_blank">
          <img src={reactLogo} className="logo react" alt="React logo" />
        </a>
      </div>
      <h1>Digit Recognizer</h1>
      <div className="card">
        <input type="file" onChange={handleFileChange} />
        <button onClick={handlePredict}>
          Ask Prediction
        </button>
        {prediction !== null && <p>Result is {prediction}</p>}
      </div>
    </>
  )
}

export default App

