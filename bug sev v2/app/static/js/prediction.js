document.addEventListener('DOMContentLoaded', () => {
    const formData = JSON.parse(sessionStorage.getItem('formData'));
    const fileContent = JSON.parse(sessionStorage.getItem('fileContent'));
    const selectedModel = sessionStorage.getItem('selectedModel');
    const predictionResult = document.getElementById('predictionResult');
    predictionResult.innerHTML = '';

            // Check if there is form data to send for prediction
            let dataToPredict = null;
            if (formData) {
                dataToPredict = formData;
            } else if (fileContent) {
                dataToPredict = fileContent;
            }

        if (!dataToPredict) {
            predictionResult.innerHTML = `<p class="error">No data available for prediction.</p>`;
            return;
        }
    
    
        fetch(`/predict?model=${selectedModel}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(dataToPredict),
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            predictionResult.innerHTML = `<p class="error">${data.error}</p>`;
        } else {
            sessionStorage.setItem('predictionData', JSON.stringify(data));
            data.forEach((item, index) => {
                const projectName = item['Project Name'];
                const resultCard = document.createElement('div');
                resultCard.className = 'result-card';
                resultCard.innerHTML = `
                    <h3>Project Name: ${projectName}</h3>
                    <p><strong>Severity:</strong> ${item.Severity}</p>
                    <p><strong>Is Buggy:</strong> ${item.isBuggy}</p>
                    <p><strong>Model Type:</strong> ${item['Model Type']}</p>
                    <p><strong>Probabilities:</strong></p>
                    <ul>
                        <li>Critical: ${item.Probabilities.Critical.toFixed(4)}</li>
                        <li>High: ${item.Probabilities.High.toFixed(4)}</li>
                        <li>Medium: ${item.Probabilities.Medium.toFixed(4)}</li>
                        <li>Low: ${item.Probabilities.Low.toFixed(4)}</li>
                    </ul>
                    <img src="${item.Graph}" alt="Probabilities Graph" class="graph">
                `;
                predictionResult.appendChild(resultCard);
            });
        }
    })
    .catch(err => {
        predictionResult.innerHTML = `<p class="error">An error occurred while processing the data.</p>`;
        console.error('Error:', err);
    });
});

function goBackToForm() {
    window.location.href = '/info';
}

function goBackToHome() {
    sessionStorage.clear();  // Clear sessionStorage when going back
    sessionStorage.setItem('introductionVisited', 'true');
    window.location.href = '/';
}

function downloadPDF() {
    const predictionData = JSON.parse(sessionStorage.getItem('predictionData'));
    if (predictionData) {
        fetch('/download_pdf', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(predictionData),
        })
        .then(response => response.blob())
        .then(blob => {
            const url = window.URL.createObjectURL(new Blob([blob]));
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', 'prediction_results.pdf');
            document.body.appendChild(link);
            link.click();
            link.parentNode.removeChild(link);
        })
        .catch(err => {
            console.error('Error downloading PDF:', err);
        });
    } else {
        alert('No prediction data available to download.');
    }
}