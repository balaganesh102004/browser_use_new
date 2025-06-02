import json
from datetime import datetime

def get_task_description(task_name, all_tasks):
    for task in all_tasks:
        if task["Task name"].lower() == task_name.lower():
            return task["Task description"]
    return "Task not found"

def render_report(history_data: dict, all_tasks: any, test_run_id: str = None) -> str:
    report_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Run Report</title>
        <link rel="stylesheet" type="text/css" href="/static/test_result.css">
        <script>
        document.addEventListener("DOMContentLoaded", function () {
            const headers = document.querySelectorAll(".accordion-header");
            headers.forEach(header => {
                header.addEventListener("click", function () {
                    const content = this.nextElementSibling;
                    content.style.display = content.style.display === "block" ? "none" : "block";
                });
            });

            const modal = document.getElementById("screenshotModal");
            const modalImg = document.getElementById("modalImage");
            const closeModal = document.getElementsByClassName("close")[0];

            document.querySelectorAll(".screenshot").forEach(img => {
                img.addEventListener("click", function () {
                    modal.style.display = "block";
                    modalImg.src = this.src;
                });
            });

            closeModal.onclick = function () {
                modal.style.display = "none";
            };

            window.onclick = function (event) {
                if (event.target === modal) {
                    modal.style.display = "none";
                }
            };
        });
        </script>
    </head>
    <body class="report-body">
    """
    report_content += """
        <h1 class="summary-title">Test Run Report</h1>
        <!-- Modal for screenshots -->
        <div id="screenshotModal" class="modal">
            <span class="close">&times;</span>
            <img id="modalImage" class="modal-content">
        </div>"""
    overall_duration = 0
    overall_status = "Success"
    start_time = None
    end_time = None
    total_tokens = 0 

    # Calculate overall duration, status, start/end times, and total tokens
    for task_name, task_data in history_data.items():
        steps = task_data.get("history", [])
        tot_duration = 0
        task_status = "Failure"
        if steps:
            last_result = steps[-1].get("result", [])
            if isinstance(last_result, list) and last_result and last_result[0].get("success", False):
                task_status = "Success"
        if task_status == "Failure":
            overall_status = "Failure"
        if steps:
            step_start = steps[0].get("metadata", {}).get("step_start_time", 0)
            step_end = steps[-1].get("metadata", {}).get("step_end_time", 0)
            if step_start:
                if start_time is None or step_start < start_time:
                    start_time = step_start
            if step_end:
                if end_time is None or step_end > end_time:
                    end_time = step_end
        for step in steps:
            metadata = step.get("metadata", {})
            start = metadata.get("step_start_time", 0)
            end = metadata.get("step_end_time", 0)
            tot_duration += round(end - start, 2)
            # Add tokens
            try:
                tokens = int(metadata.get("input_tokens", 0))
            except Exception:
                tokens = 0
            total_tokens += tokens
        overall_duration += tot_duration

    # Format start and end times
    start_time_str = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S') if start_time else "N/A"
    end_time_str = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S') if end_time else "N/A"

    # Add summary section at the top
    report_content += f"""
    <div class="summary">
        <div class="summary-field">
            <span class="summary-label">Test Run ID:</span>
            <span class="summary-value">{test_run_id if test_run_id else "N/A"}</span>
        </div>
        <div class="summary-field">
            <span class="summary-label">Status:</span>
            <span class="status-pill status-{overall_status.lower()}">{overall_status}</span>
        </div>
        <div class="summary-field">
            <span class="summary-label">Total Duration:</span>
            <span class="summary-value">{round(overall_duration, 2)} seconds</span>
        </div>
        <div class="summary-field">
            <span class="summary-label">Total Tokens:</span>
            <span class="summary-value">{total_tokens}</span>
        </div>
        <div class="summary-field">
            <span class="summary-label">Start Time:</span>
            <span class="summary-value">{start_time_str}</span>
        </div>
        <div class="summary-field">
            <span class="summary-label">End Time:</span>
            <span class="summary-value">{end_time_str}</span>
        </div>
    </div>
    """

    # Render report for each task (no nested loop)
    accordion_class = "accordion"
    for task_name, task_data in history_data.items():
        steps = task_data.get("history", [])
        tot_duration = 0
        task_status = "Failure"
        if steps:
            last_result = steps[-1].get("result", [])
            if isinstance(last_result, list) and last_result and last_result[0].get("success", False):
                task_status = "Success"

        # Calculate total duration for the task
        for step in steps:
            metadata = step.get("metadata", {})
            start = metadata.get("step_start_time", 0)
            end = metadata.get("step_end_time", 0)
            tot_duration += round(end - start, 2)

        task_description = get_task_description(task_name, all_tasks)
        # Add accordion for the task
        report_content += f"""
        <div class="{accordion_class}">
            <div class="accordion-header" onclick="toggleAccordion(this)">
                <div>
                    <strong>{task_name}:</strong> {task_description}
                </div>
                <div>
                    <span>Status:</span><span class="status-pill status-{task_status.lower()}">{task_status}</span> | <span>Total Duration: {round(tot_duration, 2)} seconds</span>
                </div>
            </div>
            <div class="accordion-content" style="display:block;">
        """

        # Add table for the task
        report_content += """
        <table>
            <tr>
                <th>Step</th>
                <th>Description</th>
                <th>Expected Result</th>
                <th>Actual Result</th>
                <th>Step Result</th>
                <th>Input Tokens</th>
                <th>Screenshot</th>
                <th>Action</th>
            </tr>
        """

        for i, step in enumerate(steps):
            model_output = step.get("model_output", {})
            current_state = model_output.get("current_state", {}) if model_output else {}
            actions = model_output.get("action", []) if model_output else []
            metadata = step.get("metadata", {})
            base64_image = step.get("state", {}).get("screenshot", "")
            result_list = step.get("result", [])

            # Step result logic: first word of next step's evaluation_previous_goal, or last result for the final step
            if i + 1 < len(steps):
                next_step = steps[i + 1]
                if isinstance(next_step, dict) and next_step.get("model_output"):
                    eval_goal = (
                        next_step["model_output"]
                        .get("current_state", {})
                        .get("evaluation_previous_goal", "N/A")
                    )
                    # Take the first word
                    step_result = classify_sentence(eval_goal)
                else:
                    step_result = "N/A"
            else:
                # For the last step, fallback to previous logic
                if isinstance(result_list, list) and result_list and isinstance(result_list[0], dict):
                    step_result = result_list[0].get("is_done", "N/A")
                    if step_result is True:
                        step_result = "Success"
                    else:
                        step_result = "Failure"
                else:
                    step_result = "N/A"

            # Action HTML
            action_html = "<ul class='action-log-list'>"
            for action in actions:
                action_type = list(action.keys())[0]
                action_data = action[action_type]
                action_html += f"<li><strong>{action_type}:</strong><ul>"
                for key, value in action_data.items():
                    action_html += f"<li><strong>{key}:</strong> {value}</li>"
                action_html += "</ul></li>"
            action_html += "</ul>"

            # Next eval logic
            if i + 1 < len(steps):
                next_step = steps[i + 1]
                if isinstance(next_step, dict) and next_step.get("model_output"):
                    next_eval = (
                        next_step["model_output"]
                        .get("current_state", {})
                        .get("evaluation_previous_goal", "N/A")
                    )
                else:
                    next_eval = "N/A"
            else:
                if task_status == "Success":
                    next_eval = "Successfully executed all the steps."
                else:
                    next_eval = "Failed to execute."

            report_content += f"""
            <tr>
                <td>{i + 1}</td>
                <td>{current_state.get("memory", "")}</td>
                <td>{current_state.get("next_goal", "")}</td>
                <td>{next_eval}</td>
                <td><span class="status-pill status-{step_result.lower()}">{step_result}</span></td>
                <td>{metadata.get("input_tokens", "")}</td>
                <td>"""
            if base64_image:
                report_content += f"""<img src="data:image/png;base64,{base64_image}" class="screenshot" alt="Step {i} screenshot">"""
            report_content += "</td>"
            report_content += f"""
                <td class="action-log"><div class="action-log-wrapper">{action_html}</div></td>
            </tr>
            """

        report_content += '</table></div></div>'

    # Close the HTML tags   
    report_content += "</body></html>"
    return report_content


def classify_sentence(text):
    success_keywords = {"success", "successfully", "completed", "done", "achieved", "returned relevant results"}
    failure_keywords = {"fail", "failed", "failure", "error", "not", "unable", "unsuccessful","retry"}

    text_lower = text.lower()

    success_found = any(word in text_lower for word in success_keywords)
    failure_found = any(word in text_lower for word in failure_keywords)

    if failure_found:
        return "Failure"
    elif success_found:
        return "Success"
    else:
        return ""
