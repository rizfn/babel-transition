# Define the ranges for alpha and gamma parameters
$alpha_values = 110, 120, 130, 140, 150, 160, 170, 180
$gamma_values = 0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0

# Fixed parameters
$max_depth = 20
$N = 1000
$L = 16
$N_rounds = 500
$mu = 0.01
$beta = 0.1
$generations = 1000

# Path to the executable
$executable = ".\beta.exe"

# Get the number of available processors and keep 8 cores free
$totalCPUs = (Get-WmiObject -Class Win32_Processor).NumberOfLogicalProcessors
$maxParallel = [Math]::Max(1, $totalCPUs - 8)
Write-Host "Running with max $maxParallel parallel jobs (total CPUs: $totalCPUs)"

# Function to run a single simulation
function Run-Simulation {
    param (
        [double]$gamma,
        [double]$alpha
    )
    
    $jobName = "g_${gamma}_a_${alpha}_d_${max_depth}"
    Write-Host "Starting job: $jobName"
    
    # Build the command line arguments (gamma, alpha, max_depth, N, L, N_rounds, mu, beta, generations)
    $args = @(
        $gamma,
        $alpha,
        $max_depth,
        $N,
        $L,
        $N_rounds,
        $mu,
        $beta,
        $generations
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