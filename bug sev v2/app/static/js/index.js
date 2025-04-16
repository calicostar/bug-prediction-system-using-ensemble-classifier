function showManualInput() {
    document.getElementById('manualInputContainer').style.display = 'flex';
    document.getElementById('uploadInputContainer').style.display = 'none';
}

function showUploadInput(type) {
    document.getElementById('manualInputContainer').style.display = 'none';
    document.getElementById('uploadInputContainer').style.display = 'flex';
    document.getElementById('uploadTitle').textContent = `Upload ${type.toUpperCase()} File`;
    document.getElementById('fileInput').accept = type === 'csv' ? '.csv' : '.jsonl';
}

function submitManualInput() {
    const form = document.getElementById('manualInputForm');
    const formData = new FormData(form);
    const formObj = {};
    formData.forEach((value, key) => formObj[key] = value);
    sessionStorage.setItem('formData', JSON.stringify(formObj));
    sessionStorage.removeItem('fileContent');  // Clear the fileContent storage to prevent conflicts
    window.location.href = '/info';
}

function submitUploadInput() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    if (!file) {
        alert('Please select a file to upload.');
        return;
    }
    const formData = new FormData();
    formData.append('file', file);

    fetch('/upload', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);  // Show error message if there is one
        } else {
            sessionStorage.setItem('fileContent', JSON.stringify(data));
            sessionStorage.removeItem('formData');  // Clear the formData storage to prevent conflicts
            window.location.href = '/info';
        }
    })
    .catch(err => {
        console.error('Error:', err);
    });
}





