# Define the ranges for alpha and gamma parameters
$alpha_values = -1, 0, 1, 2, 3, 4, 5, 6
$gamma_values = -1, 0, 1, 2, 3, 4, 5, 6

# Fixed parameters
$N = 1000
$L = 16
$N_rounds = 500
$mu = 0.01
$generations = 1000

# Path to the executable
$executable = ".\top50.exe"

# Get the number of available processors and keep 2 cores free
$totalCPUs = (Get-WmiObject -Class Win32_Processor).NumberOfLogicalProcessors
$maxParallel = [Math]::Max(1, $totalCPUs - 2)
Write-Host "Running with max $maxParallel parallel jobs (total CPUs: $totalCPUs)"

function Run-Simulation {
    param (
        [double]$gamma,
        [double]$alpha
    )
    $jobName = "g_${gamma}_a_${alpha}"
    Write-Host "Starting job: $jobName"
    $args = @(
        $gamma,
        $alpha,
        $N,
        $L,
        $N_rounds,
        $mu,
        $generations
    )
    $process = Start-Process -FilePath $executable -ArgumentList $args -NoNewWindow -PassThru
    return @{
        Process = $process
        JobName = $jobName
        StartTime = Get-Date
    }
}

$runningJobs = @()
$combinations = @()
foreach ($alpha in $alpha_values) {
    foreach ($gamma in $gamma_values) {
        $combinations += @{
            Alpha = $alpha
            Gamma = $gamma
        }
    }
}

Write-Host "Total jobs to run: $($combinations.Count)"
$completedCount = 0

foreach ($combo in $combinations) {
    while ($runningJobs.Count -ge $maxParallel) {
        $stillRunning = @()
        foreach ($job in $runningJobs) {
            if ($job.Process.HasExited) {
                $duration = (Get-Date) - $job.StartTime
                $completedCount++
                Write-Host "Job completed: $($job.JobName) - Duration: $([math]::Round($duration.TotalMinutes, 2)) minutes - Progress: $completedCount/$($combinations.Count) ($(($completedCount/$combinations.Count).ToString("P0")))"
            } else {
                $stillRunning += $job
            }
        }
        $runningJobs = $stillRunning
        if ($runningJobs.Count -ge $maxParallel) {
            Start-Sleep -Seconds 2
        }
    }
    $runningJobs += Run-Simulation -gamma $combo.Gamma -alpha $combo.Alpha
}

while ($runningJobs.Count -gt 0) {
    Start-Sleep -Seconds 2
    $stillRunning = @()
    foreach ($job in $runningJobs) {
        if ($job.Process.HasExited) {
            $duration = (Get-Date) - $job.StartTime
            $completedCount++
            Write-Host "Job completed: $($job.JobName) - Duration: $([math]::Round($duration.TotalMinutes, 2)) minutes - Progress: $completedCount/$($combinations.Count) ($(($completedCount/$combinations.Count).ToString("P0")))"
        } else {
            $stillRunning += $job
        }
    }
    $runningJobs = $stillRunning
    if ($runningJobs.Count -gt 0) {
        Write-Host "Waiting for $($runningJobs.Count) jobs to complete... ($completedCount/$($combinations.Count) done)"
    }
}

Write-Host "All jobs completed!" -ForegroundColor Green