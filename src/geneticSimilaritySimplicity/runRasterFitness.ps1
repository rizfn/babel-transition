# Define the ranges for gamma and alpha parameters
$gamma_values = -5.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0
$alpha_values = 0.0, 2.0, 4.0, 6.0, 8.0, 10.0

# Fixed parameters
$N = 1000
$L = 16
$N_rounds = 500
$mu = 0.01
$beta = 0.1
$generations = 1000
$max_depth = 5

# Path to the executable
$executable = ".\beta.exe"

# Get the number of available processors and keep 4 cores free
$totalCPUs = (Get-WmiObject -Class Win32_Processor).NumberOfLogicalProcessors
$maxParallel = [Math]::Max(1, $totalCPUs - 8)
Write-Host "Running with max $maxParallel parallel jobs (total CPUs: $totalCPUs)"

# Function to run a single simulation
function Run-Simulation {
    param (
        [double]$gamma,
        [double]$alpha
    )
    
    $jobName = "g_${gamma}_a_${alpha}"
    Write-Host "Starting job: $jobName"
    
    # Build the command line arguments
    $args = @(
        $gamma,
        $alpha,
        $N,
        $L,
        $N_rounds,
        $mu,
        $beta,
        $generations,
        $max_depth
    )
    
    # Start the process
    $process = Start-Process -FilePath $executable -ArgumentList $args -NoNewWindow -PassThru
    
    return @{
        Process = $process
        JobName = $jobName
        StartTime = Get-Date
    }
}

# Track running jobs
$runningJobs = @()

# Create all parameter combinations
$combinations = @()
foreach ($gamma in $gamma_values) {
    foreach ($alpha in $alpha_values) {
        $combinations += @{
            Gamma = $gamma
            Alpha = $alpha
        }
    }
}

Write-Host "Total jobs to run: $($combinations.Count)"
$completedCount = 0

# Process all combinations
foreach ($combo in $combinations) {
    # Wait if we have reached max parallel jobs
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
    
    # Start a new job
    $runningJobs += Run-Simulation -gamma $combo.Gamma -alpha $combo.Alpha
}

# Wait for remaining jobs to complete
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