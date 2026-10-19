param(
    [string]$SummaryPath = (Join-Path $PSScriptRoot '..\results\offline_sweep_summary_20261019.json'),
    [string]$OutputPath = (Join-Path $PSScriptRoot '..\results\figures\offline_sweep_20261019.png')
)

# 辅助绘图：只读取已有统计，不运行模型，也不改动学习者的推理代码。
# 圆点是五轮中位数，误差棒是实际最小值到最大值，不是置信区间。
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$summary = Get-Content -Raw -LiteralPath $SummaryPath | ConvertFrom-Json
$points = @($summary.points | Sort-Object requests)
if ($points.Count -ne 7) { throw 'Expected seven request-count points.' }
$OutputPath = [IO.Path]::GetFullPath($OutputPath)
New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($OutputPath)) -Force | Out-Null

$bitmap = [Drawing.Bitmap]::new(1550, 720)
$graphics = [Drawing.Graphics]::FromImage($bitmap)
$graphics.Clear([Drawing.Color]::White)
$graphics.SmoothingMode = [Drawing.Drawing2D.SmoothingMode]::AntiAlias
$graphics.TextRenderingHint = [Drawing.Text.TextRenderingHint]::AntiAliasGridFit
$font = [Drawing.Font]::new('Segoe UI', 18, [Drawing.FontStyle]::Regular, [Drawing.GraphicsUnit]::Pixel)
$smallFont = [Drawing.Font]::new('Segoe UI', 16, [Drawing.FontStyle]::Regular, [Drawing.GraphicsUnit]::Pixel)
$titleFont = [Drawing.Font]::new('Segoe UI', 30, [Drawing.FontStyle]::Bold, [Drawing.GraphicsUnit]::Pixel)
$panelFont = [Drawing.Font]::new('Segoe UI', 21, [Drawing.FontStyle]::Bold, [Drawing.GraphicsUnit]::Pixel)
$ink = [Drawing.SolidBrush]::new([Drawing.ColorTranslator]::FromHtml('#253449'))
$muted = [Drawing.SolidBrush]::new([Drawing.ColorTranslator]::FromHtml('#53657b'))
$blueBrush = [Drawing.SolidBrush]::new([Drawing.ColorTranslator]::FromHtml('#1976b9'))
$bluePen = [Drawing.Pen]::new($blueBrush.Color, 3)
$rangePen = [Drawing.Pen]::new([Drawing.ColorTranslator]::FromHtml('#8dbbd9'), 2)
$gridPen = [Drawing.Pen]::new([Drawing.ColorTranslator]::FromHtml('#e3e9ef'), 1)
$axisPen = [Drawing.Pen]::new([Drawing.ColorTranslator]::FromHtml('#738297'), 1)
$limitPen = [Drawing.Pen]::new([Drawing.ColorTranslator]::FromHtml('#a1acb9'), 1)
$limitPen.DashStyle = [Drawing.Drawing2D.DashStyle]::Dash

function Draw-Text($Text, $X, $Y, $TextFont, $Brush) {
    $graphics.DrawString([string]$Text, $TextFont, $Brush, [single]$X, [single]$Y)
}

function Draw-Panel($Left, $Top, $Width, $Height, $YMax, $YStep, $Mode, $Title) {
    Draw-Text $Title $Left ($Top - 42) $panelFont $ink
    for ($tick = 0; $tick -le $YMax; $tick += $YStep) {
        $y = $Top + $Height * (1 - $tick / $YMax)
        $graphics.DrawLine($gridPen, [single]$Left, [single]$y, [single]($Left + $Width), [single]$y)
        Draw-Text $tick ($Left - 52) ($y - 12) $smallFont $muted
    }
    $graphics.DrawLine($axisPen, [single]$Left, [single]$Top, [single]$Left, [single]($Top + $Height))
    $limitX = $Left + 4 * $Width / 6
    $graphics.DrawLine($limitPen, [single]$limitX, [single]$Top, [single]$limitX, [single]($Top + $Height))
    $previous = $null
    for ($i = 0; $i -lt $points.Count; $i++) {
        $point = $points[$i]
        $x = $Left + $i * $Width / 6
        if ($Mode -eq 'throughput') {
            $median = $point.output_tps_median
            $low = $point.output_tps_min
            $high = $point.output_tps_max
            $label = '{0:0}' -f $median
        } else {
            $median = $point.median_s
            $low = $point.min_s
            $high = $point.max_s
            $label = '{0:0.00}' -f $median
        }
        $y = $Top + $Height * (1 - $median / $YMax)
        $yLow = $Top + $Height * (1 - $low / $YMax)
        $yHigh = $Top + $Height * (1 - $high / $YMax)
        if ($null -ne $previous) {
            $graphics.DrawLine($bluePen, [single]$previous.X, [single]$previous.Y, [single]$x, [single]$y)
        }
        $graphics.DrawLine($rangePen, [single]$x, [single]$yLow, [single]$x, [single]$yHigh)
        $graphics.DrawLine($rangePen, [single]($x - 6), [single]$yLow, [single]($x + 6), [single]$yLow)
        $graphics.DrawLine($rangePen, [single]($x - 6), [single]$yHigh, [single]($x + 6), [single]$yHigh)
        $graphics.FillEllipse($blueBrush, [single]($x - 5), [single]($y - 5), [single]10, [single]10)
        Draw-Text $label ($x - 20) ($yHigh - 27) $smallFont $ink
        Draw-Text $point.requests ($x - 10) ($Top + $Height + 12) $font $ink
        $previous = @{X=$x; Y=$y}
    }
    Draw-Text 'Requests per generate call (N; log2 scale)' ($Left + 95) ($Top + $Height + 49) $font $muted
}

try {
    Draw-Text 'Offline throughput sweep - fixed engine configuration' 50 22 $titleFont $ink
    Draw-Text 'Qwen2.5-1.5B-Instruct | BF16 | ISL/OSL = 512/128 | max_num_seqs = 16 | token budget = 2048' 50 70 $font $muted
    Draw-Panel 100 160 610 390 1000 200 'throughput' 'Output throughput (tokens/s)'
    Draw-Panel 850 160 600 390 10 2 'time' 'Batch completion time (s)'
    Draw-Text 'Markers: median of 5 rounds; whiskers: observed min-max, NOT a confidence interval. Warmup: 2 rounds per N.' 50 638 $smallFont $muted
    Draw-Text 'Dashed line: configured max_num_seqs=16, NOT proof of GPU saturation. Prefix caching off; identical synthetic prompts.' 50 669 $smallFont $muted
    $bitmap.Save($OutputPath, [Drawing.Imaging.ImageFormat]::Png)
} finally {
    $graphics.Dispose()
    $bitmap.Dispose()
    foreach ($resource in @($font, $smallFont, $titleFont, $panelFont, $ink, $muted, $blueBrush, $bluePen, $rangePen, $gridPen, $axisPen, $limitPen)) { $resource.Dispose() }
}
Write-Output $OutputPath
