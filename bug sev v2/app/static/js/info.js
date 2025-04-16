document.addEventListener('DOMContentLoaded', () => {
    const formData = JSON.parse(sessionStorage.getItem('formData'));
    const fileContent = JSON.parse(sessionStorage.getItem('fileContent'));
    const infoContent = document.getElementById('infoContent');
    infoContent.innerHTML = '';

    if (formData) {
        const card = document.createElement('div');
        card.className = 'info-card';
        card.innerHTML = `
            <h3 class="bold-large-title">Project Information</h3>
            <pre class="bold-large-text">'project_name': ["${formData.project_name}"],</pre>
            <pre class="bold-large-text">'project_version': ["${formData.project_version}"],</pre>
            <pre class="bold-large-text">'label': [${formData.label}],</pre>
            <pre class="bold-large-text">'code': [\`${formData.code}\`],</pre>
            <pre class="bold-large-text">'code_comment': ["${formData.code_comment}"],</pre>
            <pre class="bold-large-text">'code_no_comment': ["${formData.code_no_comment}"],</pre>
            <pre class="bold-large-text">'lc': [${formData.lc}],</pre>
            <pre class="bold-large-text">'pi': [${formData.pi}],</pre>
            <pre class="bold-large-text">'ma': [${formData.ma}],</pre>
            <pre class="bold-large-text">'nbd': [${formData.nbd}],</pre>
            <pre class="bold-large-text">'ml': [${formData.ml}],</pre>
            <pre class="bold-large-text">'d': [${formData.d}],</pre>
            <pre class="bold-large-text">'mi': [${formData.mi}],</pre>
            <pre class="bold-large-text">'fo': [${formData.fo}],</pre>
            <pre class="bold-large-text">'r': [${formData.r}],</pre>
            <pre class="bold-large-text">'e': [${formData.e}]</pre>
        `;
        infoContent.appendChild(card);
    } else if (fileContent) {
        const p = document.createElement('p');
        p.textContent = 'File uploaded successfully.';
        infoContent.appendChild(p);

        fileContent.forEach(item => {
            const card = document.createElement('div');
            card.className = 'info-card';
            card.innerHTML = `
                <h3 class="bold-large-title">Project Information</h3>
                <pre class="bold-large-text">'project_name': ["${item.project_name}"],</pre>
                <pre class="bold-large-text">'project_version': ["${item.project_version}"],</pre>
                <pre class="bold-large-text">'label': [${item.label}],</pre>
                <pre class="bold-large-text">'code': [\`${item.code}\`],</pre>
                <pre class="bold-large-text">'code_comment': ["${item.code_comment}"],</pre>
                <pre class="bold-large-text">'code_no_comment': ["${item.code_no_comment}"],</pre>
                <pre class="bold-large-text">'lc': [${item.lc}],</pre>
                <pre class="bold-large-text">'pi': [${item.pi}],</pre>
                <pre class="bold-large-text">'ma': [${item.ma}],</pre>
                <pre class="bold-large-text">'nbd': [${item.nbd}],</pre>
                <pre class="bold-large-text">'ml': [${item.ml}],</pre>
                <pre class="bold-large-text">'d': [${item.d}],</pre>
                <pre class="bold-large-text">'mi': [${item.mi}],</pre>
                <pre class="bold-large-text">'fo': [${item.fo}],</pre>
                <pre class="bold-large-text">'r': [${item.r}],</pre>
                <pre class="bold-large-text">'e': [${item.e}]</pre>
            `;
            infoContent.appendChild(card);
        });
    } else {
        const p = document.createElement('p');
        p.textContent = 'No data available.';
        infoContent.appendChild(p);
    }

});

function goBackToForm() {
    window.location.href = '/';
}

function goToPredictionPage() {
    window.location.href = '/prediction';
}

function predictWithModel(modelType) {
    sessionStorage.setItem('selectedModel', modelType);
    window.location.href = '/prediction';
}
