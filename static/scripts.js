function openEditModalFromButton(button) {
    const taskId = button.getAttribute('data-id');
    const name = button.getAttribute('data-name');
    const description = button.getAttribute('data-description');
    const tags = button.getAttribute('data-tags');

    document.getElementById('edit-modal').style.display = 'flex';
    document.getElementById('edit-task-id').value = taskId;
    document.getElementById('edit-task-name').value = name;
    document.getElementById('edit-task-description').value = description;
    document.getElementById('edit-task-tags').value = tags;
}

function closeEditModal() {
    document.getElementById('edit-modal').style.display = 'none';
}

function generateField(id) {
    document.getElementById(id).value = "Generated " + id;
}

function openCreateModal() {
    document.getElementById('create-modal').style.display = 'flex';
}

function closeCreateModal() {
    document.getElementById('create-modal').style.display = 'none';
}

window.onclick = function(event) {
    const modal = document.getElementById('create-modal');
    if (event.target === modal) {
        modal.style.display = "none";
    }
};

document.addEventListener("DOMContentLoaded", function() {
    // Edit form submit
    const editForm = document.getElementById("edit-form");
    if (editForm) {
        editForm.addEventListener("submit", function (e) {
            e.preventDefault();
            const taskId = document.getElementById("edit-task-id").value;
            this.action = "/update/" + taskId;
            this.submit();
        });
    }

    // Search functionality
    const searchInput = document.getElementById('task-search');
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            const query = this.value.trim().toLowerCase();
            const cards = document.querySelectorAll('.task-card');
            cards.forEach(card => {
                const name = card.querySelector('h3').textContent.toLowerCase();
                const desc = card.querySelector('p').textContent.toLowerCase();
                const tags = card.querySelector('p strong') 
                    ? card.querySelector('p strong').nextSibling.textContent.toLowerCase() 
                    : '';
                if ( name.includes(query) || desc.includes(query) || tags.includes(query)) {
                    card.style.display = '';
                } else {
                    card.style.display = 'none';
                }
            });
        });
    }

    // Generate Selected Tasks
    const generateBtn = document.getElementById('generate-selected-btn');
    if (generateBtn) {
        generateBtn.addEventListener('click', function() {
            const checked = document.querySelectorAll('.task-checkbox:checked');
            if (checked.length === 0) {
                alert("Please select at least one task.");
                return;
            }
            const taskNames = Array.from(checked).map(cb => cb.value);

            // Show loader
            const loader = document.getElementById('loader-overlay');
            const loaderText = document.getElementById('loader-text');
            loaderText.textContent = `Executing ${taskNames.length} task${taskNames.length > 1 ? 's' : ''}...`;
            loader.style.display = 'flex';

            // Open the report window immediately to avoid popup blockers
            const reportWindow = window.open('', 'Test Report');
            reportWindow.document.write(`
                <html>
                <head>
                    <title>Test Report</title>
                    <link rel="stylesheet" href="${location.origin}/static/styles.css">
                </head>
                <body>
                    <div id="loader-overlay" class="loader-overlay" style="display:flex;">
                        <div class="loader-content">
                            <img src="${location.origin}/static/Loading.gif" alt="Loading..." class="loader-img">
                            <div id="loader-text" class="loader-text">
                                Executing ${taskNames.length} task${taskNames.length > 1 ? 's' : ''}...
                            </div>
                        </div>
                    </div>
                    <h1>Live Logs</h1>
                    <div id="log"></div>

                    <script>
                        const logDiv = document.getElementById("log");
                        const eventSource = new EventSource("/stream");

                        eventSource.onmessage = function(event) {
                        logDiv.innerHTML += event.data + "<br>";
                        logDiv.scrollTop = logDiv.scrollHeight;
                        };
                    </script>
                </body>
                </html>
            `);

            // Prepare form data
            const formData = new FormData();
            taskNames.forEach(name => formData.append('tasks[]', name));
            // Send POST to /run and open result in the already opened tab
            fetch('/run', {
                method: 'POST',
                body: formData
            })
            .then(response => response.text())
            .then(html => {
                loader.style.display = 'none';
                reportWindow.document.open();
                reportWindow.document.write(html);
                reportWindow.document.title = "Test Report";
                reportWindow.document.close();
            })
            .catch(err => {
                loader.style.display = 'none';
                reportWindow.document.body.innerHTML = "<h2>Error generating report.</h2><pre>" + err + "</pre>";
            });
        });
    }

    // --- Run Selected Button ---
    const runSelectedBtn = document.getElementById('run-selected-btn');
    if (runSelectedBtn) {
        runSelectedBtn.addEventListener('click', function() {
            const checked = document.querySelectorAll('.task-checkbox:checked');
            if (checked.length === 0) {
                alert("Please select at least one task.");
                return;
            }
            const taskNames = Array.from(checked).map(cb => cb.value);

            // Show loader
            const loader = document.getElementById('loader-overlay');
            const loaderText = document.getElementById('loader-text');
            loaderText.textContent = `Executing ${taskNames.length} task${taskNames.length > 1 ? 's' : ''}...`;
            loader.style.display = 'flex';

            // Open the report window immediately to avoid popup blockers
            const reportWindow = window.open('', 'Test Report');
            reportWindow.document.write(`
                <html>
                <head>
                    <title>Test Report</title>
                    <link rel="stylesheet" href="${location.origin}/static/styles.css">
                </head>
                <body>
                    <div id="loader-overlay" class="loader-overlay" style="display:flex;">
                        <div class="loader-content">
                            <img src="${location.origin}/static/Loading.gif" alt="Loading..." class="loader-img">
                            <div id="loader-text" class="loader-text">
                                Executing ${taskNames.length} task${taskNames.length > 1 ? 's' : ''}...
                            </div>
                        </div>
                    </div>
                    <h1>Live Logs</h1>
                    <div id="log"></div>

                    <script>
                        const logDiv = document.getElementById("log");
                        const eventSource = new EventSource("/stream");

                        eventSource.onmessage = function(event) {
                        logDiv.innerHTML += event.data + "<br>";
                        logDiv.scrollTop = logDiv.scrollHeight;
                        };
                    </script>
                </body>
                </html>
            `);

            // Prepare form data
            const formData = new FormData();
            taskNames.forEach(name => formData.append('tasks[]', name));

            fetch('/run', {
                method: 'POST',
                body: formData
            })
            .then(response => response.text())
            .then(html => {
                loader.style.display = 'none';
                reportWindow.document.open();
                reportWindow.document.write(html);
                reportWindow.document.title = "Test Report";
                reportWindow.document.close();
            })
            .catch(err => {
                loader.style.display = 'none';
                reportWindow.document.body.innerHTML = "<h2>Error generating report.</h2><pre>" + err + "</pre>";
            });
        });
    }

    // --- Run All Button ---
    const runAllBtn = document.getElementById('run-all-btn');
    if (runAllBtn) {
        runAllBtn.addEventListener('click', function() {
            const allCheckboxes = document.querySelectorAll('.task-checkbox');
            const taskNames = Array.from(allCheckboxes).map(cb => cb.value);

            if (taskNames.length === 0) {
                alert("No tasks available.");
                return;
            }

            // Show loader
            const loader = document.getElementById('loader-overlay');
            const loaderText = document.getElementById('loader-text');
            loaderText.textContent = `Executing all ${taskNames.length} task${taskNames.length > 1 ? 's' : ''}...`;
            loader.style.display = 'flex';

            // Open the report window immediately to avoid popup blockers
            const reportWindow = window.open('', 'Test Report');
            reportWindow.document.write(`
                <html>
                <head>
                    <title>Test Report</title>
                    <link rel="stylesheet" href="${location.origin}/static/styles.css">
                </head>
                <body>
                    <div id="loader-overlay" class="loader-overlay" style="display:flex;">
                        <div class="loader-content">
                            <img src="${location.origin}/static/Loading.gif" alt="Loading..." class="loader-img">
                            <div id="loader-text" class="loader-text">
                                Executing all ${taskNames.length} task${taskNames.length > 1 ? 's' : ''}...
                            </div>
                        </div>
                    </div>
                    <h1>Live Logs</h1>
                    <div id="log"></div>

                    <script>
                        const logDiv = document.getElementById("log");
                        const eventSource = new EventSource("/stream");

                        eventSource.onmessage = function(event) {
                        logDiv.innerHTML += event.data + "<br>";
                        logDiv.scrollTop = logDiv.scrollHeight;
                        };
                    </script>
                </body>
                </html>
            `);

            // Prepare form data
            const formData = new FormData();
            taskNames.forEach(name => formData.append('tasks[]', name));

            fetch('/run', {
                method: 'POST',
                body: formData
            })
            .then(response => response.text())
            .then(html => {
                loader.style.display = 'none';
                reportWindow.document.open();
                reportWindow.document.write(html);
                reportWindow.document.title = "Test Report";
                reportWindow.document.close();
            })
            .catch(err => {
                loader.style.display = 'none';
                reportWindow.document.body.innerHTML = "<h2>Error generating report.</h2><pre>" + err + "</pre>";
            });
        });
    }

    document.querySelectorAll('.task-checkbox').forEach(cb => {
        cb.addEventListener('change', function() {
            const count = document.querySelectorAll('.task-checkbox:checked').length;
            document.getElementById('selected-count').textContent = `Tasks Selected: ${count}`;
        });
    });
});

// --- Generate Button (per task) ---
function generateTask(taskId) {
    // Find the task name using the checkbox value or data attribute
    const taskCard = document.querySelector(`.generate-button[onclick="generateTask('${taskId}')"]`).closest('.task-card');
    const taskName = taskCard.querySelector('.task-checkbox').value;

    // Show loader
    const loader = document.getElementById('loader-overlay');
    const loaderText = document.getElementById('loader-text');
    loaderText.textContent = "Generating task...";
    loader.style.display = 'flex';

    // Open the report window immediately to avoid popup blockers
    const reportWindow = window.open('', 'Task Result');
    reportWindow.document.write(`
        <html>
        <head>
            <title>Task Result</title>
            <link rel="stylesheet" href="${location.origin}/static/styles.css">
            <style>
                #log {
                width: 100%;
                height: 400px;
                background: #000;
                color: #0f0;
                overflow-y: scroll;
                padding: 10px;
                font-family: monospace;
                }
            </style>
        </head>
        <body>
            <div id="loader-overlay" class="loader-overlay" style="display:flex;">
                <div class="loader-content">
                    <img src="${location.origin}/static/Loading.gif" alt="Loading..." class="loader-img">
                    <div id="loader-text" class="loader-text">
                        Performing the task... Please wait.
                    </div>
                </div>
            </div>
            <h1>Live Logs</h1>
            <div id="log"></div>

            <script>
                const logDiv = document.getElementById("log");
                const eventSource = new EventSource("/stream");

                eventSource.onmessage = function(event) {
                logDiv.innerHTML += event.data + "<br>";
                logDiv.scrollTop = logDiv.scrollHeight;
                };
            </script>
        </body>
        </html>
    `);

    // Prepare form data
    const formData = new FormData();
    formData.append('tasks[]', taskName);

    fetch('/run', {
        method: 'POST',
        body: formData
    })
    .then(response => response.text())
    .then(html => {
        loader.style.display = 'none';
        reportWindow.document.open();
        reportWindow.document.write(html);
        reportWindow.document.title = "Task Result";
        reportWindow.document.close();
    })
    .catch(err => {
        loader.style.display = 'none';
        reportWindow.document.body.innerHTML = "<h2>Error generating task.</h2><pre>" + err + "</pre>";
    });
}

window.generateTask = function(buttonElement) {
    // Find the task name from the checkbox in the same card
    const taskCard = buttonElement.closest('.task-card');
    const checkbox = taskCard.querySelector('.task-checkbox');
    if (!checkbox) {
        alert("Task checkbox not found.");
        return;
    }
    const taskName = checkbox.value;

    // Show loader
    const loader = document.getElementById('loader-overlay');
    const loaderText = document.getElementById('loader-text');
    loaderText.textContent = "Generating task...";
    loader.style.display = 'flex';

    // Open the report window immediately to avoid popup blockers
    const reportWindow = window.open('', 'Task Result');
    reportWindow.document.write(`
        <html>
        <head>
            <title>Task Result</title>
            <link rel="stylesheet" href="${location.origin}/static/styles.css">
        </head>
        <body>
            <div id="loader-overlay" class="loader-overlay" style="display:flex;">
                <div class="loader-content">
                    <img src="${location.origin}/static/Loading.gif" alt="Loading..." class="loader-img">
                    <div id="loader-text" class="loader-text">
                        Performing the task... Please wait.
                    </div>
                </div>
            </div>
            <h1>Live Logs</h1>
            <div id="log"></div>

            <script>
                const logDiv = document.getElementById("log");
                const eventSource = new EventSource("/stream");

                eventSource.onmessage = function(event) {
                logDiv.innerHTML += event.data + "<br>";
                logDiv.scrollTop = logDiv.scrollHeight;
                };
            </script>
        </body>
        </html>
    `);

    // Prepare form data with a single task in the array
    const formData = new FormData();
    formData.append('tasks[]', taskName);

    fetch('/run', {
        method: 'POST',
        body: formData
    })
    .then(response => response.text())
    .then(html => {
        loader.style.display = 'none';
        reportWindow.document.open();
        reportWindow.document.write(html);
        reportWindow.document.title = "Task Result";
        reportWindow.document.close();
    })
    .catch(err => {
        loader.style.display = 'none';
        reportWindow.document.body.innerHTML = "<h2>Error generating task.</h2><pre>" + err + "</pre>";
    });
};