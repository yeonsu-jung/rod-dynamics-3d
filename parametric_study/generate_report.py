import csv
import json
import os

import sys

CSV_PATH = "build-headless/long_sim_profile.csv"
HTML_PATH = "analysis_report.html"
DT = 0.001

def main():
    global CSV_PATH, HTML_PATH
    if len(sys.argv) > 1:
        CSV_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        HTML_PATH = sys.argv[2]
        
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return

    frames = []
    ke = []
    contacts = []
    soft_pe = []
    gyration = []
    reldisp = []

    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                fr = int(row['frame'])
                frames.append(fr * DT) # Time in seconds
                ke.append(float(row['KE']))
                contacts.append(int(row['contacts']))
                if 'soft_PE' in row:
                    soft_pe.append(float(row['soft_PE']))
                else:
                    soft_pe.append(0.0)
                if 'gyration_sq' in row:
                    gyration.append(float(row['gyration_sq']))
                else:
                    gyration.append(0.0)
                if 'reldisp_sq' in row:
                    reldisp.append(float(row['reldisp_sq']))
                else:
                    reldisp.append(0.0)
            except ValueError:
                continue

    # Create HTML with Chart.js
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Simulation Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: sans-serif; margin: 20px; }}
        .chart-container {{ width: 800px; margin: 20px auto; }}
    </style>
</head>
<body>
    <h1>Simulation Analysis Report</h1>
    <p>Data source: {CSV_PATH}</p>
    <p>Total frames: {len(frames)}</p>

    <div class="chart-container">
        <canvas id="keChart"></canvas>
    </div>
    <div class="chart-container">
        <canvas id="contactsChart"></canvas>
    </div>
    <div class="chart-container">
        <canvas id="peChart"></canvas>
    </div>
    <div class="chart-container">
        <canvas id="gyrationChart"></canvas>
    </div>
    <div class="chart-container">
        <canvas id="reldispChart"></canvas>
    </div>

    <script>
        const timeData = {json.dumps(frames)};
        const keData = {json.dumps(ke)};
        const contactsData = {json.dumps(contacts)};
        const peData = {json.dumps(soft_pe)};
        const gyrationData = {json.dumps(gyration)};
        const reldispData = {json.dumps(reldisp)};

        function createChart(id, label, data, color) {{
            const ctx = document.getElementById(id).getContext('2d');
            new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: timeData,
                    datasets: [{{
                        label: label,
                        data: data,
                        borderColor: color,
                        borderWidth: 1,
                        pointRadius: 0,
                        fill: false
                    }}]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        x: {{ title: {{ display: true, text: 'Time (s)' }} }},
                        y: {{ title: {{ display: true, text: label }} }}
                    }},
                    interaction: {{
                        mode: 'index',
                        intersect: false
                    }}
                }}
            }});
        }}

        createChart('keChart', 'Kinetic Energy', keData, 'rgb(75, 192, 192)');
        createChart('contactsChart', 'Number of Contacts', contactsData, 'rgb(255, 99, 132)');
        createChart('peChart', 'Soft Potential Energy', peData, 'rgb(54, 162, 235)');
        createChart('gyrationChart', 'Gyration Radius Sq', gyrationData, 'rgb(153, 102, 255)');
        createChart('reldispChart', 'Relative Displacement Sq', reldispData, 'rgb(255, 159, 64)');
    </script>
</body>
</html>
    """

    with open(HTML_PATH, 'w') as f:
        f.write(html_content)
    
    print(f"Report generated: {HTML_PATH}")

if __name__ == "__main__":
    main()
