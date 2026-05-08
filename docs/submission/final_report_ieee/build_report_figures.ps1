Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Drawing

$ReportDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ReportDir "..\..\..")
$AssetDir = Join-Path $ReportDir "assets"
New-Item -ItemType Directory -Force -Path $AssetDir | Out-Null

$SystemComparisonPath = Join-Path $RepoRoot "outputs\final\tables\system_comparison.csv"
$RepairSalvagePath = Join-Path $RepoRoot "outputs\final\tables\repair_salvage.csv"
$DistractorPath = Join-Path $RepoRoot "outputs\final\tables\generated_distractor_metrics.csv"

$SystemRows = Import-Csv -Path $SystemComparisonPath
$RepairRows = Import-Csv -Path $RepairSalvagePath
$DistractorRows = Import-Csv -Path $DistractorPath
$InvariantCulture = [System.Globalization.CultureInfo]::InvariantCulture

function Get-Number($Row, [string]$Name) {
    if ($null -eq $Row) { return 0.0 }
    $prop = $Row.PSObject.Properties | Where-Object { $_.Name -eq $Name } | Select-Object -First 1
    if ($null -eq $prop) { return 0.0 }
    $value = $prop.Value
    if ($null -eq $value -or $value -eq "") { return 0.0 }
    return [double]$value
}

function Get-Percent($Row, [string]$Name) {
    return (Get-Number $Row $Name) * 100.0
}

function Format-Percent([double]$Value) {
    return [string]::Format($InvariantCulture, "{0:0.0}%", $Value)
}

function Format-Count([double]$Value) {
    return [string]::Format($InvariantCulture, "{0:0}", $Value)
}

function Get-SystemRow([string]$Dataset, [string]$System, [string]$ModelSize = "3b") {
    return $SystemRows |
        Where-Object { $_.dataset -eq $Dataset -and $_.system -eq $System -and $_.model_size -eq $ModelSize } |
        Select-Object -First 1
}

function Get-RepairRow([string]$Dataset, [string]$ModelSize = "3b") {
    return $RepairRows |
        Where-Object { $_.dataset -eq $Dataset -and $_.system -eq "repair_plus_verifier" -and $_.model_size -eq $ModelSize } |
        Select-Object -First 1
}

function Get-DistractorRow([string]$Dataset) {
    return $DistractorRows |
        Where-Object { $_.dataset -eq $Dataset -and $_.system -eq "repair_plus_verifier" } |
        Select-Object -First 1
}

function New-Brush([string]$Hex) {
    return New-Object System.Drawing.SolidBrush ([System.Drawing.ColorTranslator]::FromHtml($Hex))
}

function New-Pen([string]$Hex, [float]$Width = 1.0) {
    return New-Object System.Drawing.Pen ([System.Drawing.ColorTranslator]::FromHtml($Hex)), $Width
}

function Draw-Text($Graphics, [string]$Text, $Font, $Brush, [float]$X, [float]$Y, [float]$W, [float]$H, [string]$Align = "Near", [string]$VAlign = "Near") {
    $format = New-Object System.Drawing.StringFormat
    $format.Alignment = [System.Drawing.StringAlignment]::$Align
    $format.LineAlignment = [System.Drawing.StringAlignment]::$VAlign
    $rect = New-Object System.Drawing.RectangleF($X, $Y, $W, $H)
    $Graphics.DrawString($Text, $Font, $Brush, $rect, $format)
    $format.Dispose()
}

function Save-Chart($Bitmap, $Graphics, [string]$Path) {
    $Graphics.Flush()
    $Bitmap.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
    $Graphics.Dispose()
    $Bitmap.Dispose()
}

function New-ChartCanvas([int]$Width, [int]$Height) {
    $bitmap = New-Object System.Drawing.Bitmap($Width, $Height)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit
    $graphics.Clear([System.Drawing.Color]::White)
    return @{ Bitmap = $bitmap; Graphics = $graphics }
}

function Draw-FooterAxis($Graphics, [float]$X, [float]$Y, [float]$W, [float]$MaxValue, $Font, $Brush, [bool]$ShowPercent) {
    $gridPen = New-Object System.Drawing.Pen ([System.Drawing.ColorTranslator]::FromHtml("#E4E7EC")), 1
    $axisPen = New-Object System.Drawing.Pen ([System.Drawing.ColorTranslator]::FromHtml("#667085")), 1.5
    $ticks = if ($ShowPercent) { @(0, 25, 50, 75, 100) } else { @(0, ($MaxValue * 0.25), ($MaxValue * 0.5), ($MaxValue * 0.75), $MaxValue) }
    foreach ($tick in $ticks) {
        if ($tick -gt $MaxValue) { continue }
        $tx = $X + ($W * $tick / $MaxValue)
        $Graphics.DrawLine($gridPen, $tx, 118, $tx, $Y)
        $label = if ($ShowPercent) { Format-Percent ([double]$tick) } else { Format-Count ([double]$tick) }
        Draw-Text $Graphics $label $Font $Brush ($tx - 40) ($Y + 10) 80 26 "Center" "Near"
    }
    $Graphics.DrawLine($axisPen, $X, $Y, ($X + $W), $Y)
    $gridPen.Dispose()
    $axisPen.Dispose()
}

function Draw-HorizontalMetricChart([string]$OutputName, [string]$Title, [string]$Subtitle, [object[]]$Rows, [float]$MaxValue, [string]$XAxisLabel) {
    $width = 1500
    $height = 860
    $canvas = New-ChartCanvas $width $height
    $g = $canvas.Graphics

    $ink = New-Brush "#101828"
    $muted = New-Brush "#667085"
    $labelBrush = New-Brush "#344054"
    $gridBrush = New-Brush "#F2F4F7"
    $zeroBrush = New-Brush "#027A48"
    $riskBrush = New-Brush "#B42318"

    $titleFont = New-Object System.Drawing.Font("Segoe UI Semibold", 34)
    $subtitleFont = New-Object System.Drawing.Font("Segoe UI", 19)
    $axisFont = New-Object System.Drawing.Font("Segoe UI", 16)
    $labelFont = New-Object System.Drawing.Font("Segoe UI Semibold", 18)
    $smallFont = New-Object System.Drawing.Font("Segoe UI", 15)
    $valueFont = New-Object System.Drawing.Font("Segoe UI Semibold", 17)

    Draw-Text $g $Title $titleFont $ink 54 34 ($width - 108) 48 "Near" "Near"
    Draw-Text $g $Subtitle $subtitleFont $muted 56 83 ($width - 112) 34 "Near" "Near"

    $left = 360.0
    $right = 106.0
    $top = 138.0
    $bottom = 116.0
    $plotWidth = $width - $left - $right
    $plotHeight = $height - $top - $bottom
    $rowGap = $plotHeight / [Math]::Max(1, $Rows.Count)
    $barHeight = [Math]::Min(54.0, $rowGap * 0.56)
    $baseY = $top + ($rowGap - $barHeight) / 2
    $showPercent = $XAxisLabel -ne "Examples"

    $baseline = $top + $plotHeight
    Draw-FooterAxis $g $left $baseline $plotWidth $MaxValue $axisFont $muted $showPercent

    for ($i = 0; $i -lt $Rows.Count; $i++) {
        $row = $Rows[$i]
        $y = $baseY + ($i * $rowGap)
        $value = [double]$row.Value
        $barWidth = [Math]::Max(2.0, $plotWidth * $value / $MaxValue)
        $barBrush = New-Brush $row.Color
        $bgRect = New-Object System.Drawing.RectangleF($left, $y, $plotWidth, $barHeight)
        $barRect = New-Object System.Drawing.RectangleF($left, $y, $barWidth, $barHeight)
        $g.FillRectangle($gridBrush, $bgRect)
        $g.FillRectangle($barBrush, $barRect)

        Draw-Text $g $row.Label $labelFont $labelBrush 56 ($y - 2) 282 ($barHeight + 4) "Near" "Center"
        $valueLabel = if ($showPercent) { Format-Percent $value } else { Format-Count $value }
        Draw-Text $g $valueLabel $valueFont $ink ($left + $barWidth + 12) ($y - 2) 118 ($barHeight + 4) "Near" "Center"

        if ($row.ContainsKey("Tag")) {
            $tagColor = if ($row.Tag -match "^UNS" -and $row.Tag -match "0\.0") {
                $zeroBrush
            } elseif ($row.Tag -match "^UNS") {
                $riskBrush
            } else {
                $muted
            }
            Draw-Text $g $row.Tag $smallFont $tagColor ($left + $plotWidth - 164) ($y - 2) 160 ($barHeight + 4) "Far" "Center"
        }

        $barBrush.Dispose()
    }

    Draw-Text $g $XAxisLabel $axisFont $muted $left ($height - 48) $plotWidth 28 "Center" "Near"

    foreach ($obj in @($ink, $muted, $labelBrush, $gridBrush, $zeroBrush, $riskBrush, $titleFont, $subtitleFont, $axisFont, $labelFont, $smallFont, $valueFont)) {
        $obj.Dispose()
    }
    Save-Chart $canvas.Bitmap $g (Join-Path $AssetDir $OutputName)
}

$asqaBaseline = Get-SystemRow "asqa" "baseline"
$asqaGateVerifier = Get-SystemRow "asqa" "gate_plus_verifier"
$asqaRepair = Get-SystemRow "asqa" "repair_plus_verifier" "3b"
$financeBaseline = Get-SystemRow "finance" "baseline"
$financeGateVerifier = Get-SystemRow "finance" "gate_plus_verifier"
$financeRepair = Get-SystemRow "finance" "repair_plus_verifier" "3b"

$safetyRows = @(
    @{ Label = "ASQA baseline"; Value = Get-Percent $asqaBaseline "answer_coverage"; Color = "#98A2B3"; Tag = ("UNS " + (Format-Percent (Get-Percent $asqaBaseline "unsupported_non_abstained_rate"))) },
    @{ Label = "ASQA gate + verifier"; Value = Get-Percent $asqaGateVerifier "answer_coverage"; Color = "#6172F3"; Tag = ("UNS " + (Format-Percent (Get-Percent $asqaGateVerifier "unsupported_non_abstained_rate"))) },
    @{ Label = "ASQA repair + verifier"; Value = Get-Percent $asqaRepair "answer_coverage"; Color = "#12B76A"; Tag = ("UNS " + (Format-Percent (Get-Percent $asqaRepair "unsupported_non_abstained_rate"))) },
    @{ Label = "Finance baseline"; Value = Get-Percent $financeBaseline "answer_coverage"; Color = "#98A2B3"; Tag = ("UNS " + (Format-Percent (Get-Percent $financeBaseline "unsupported_non_abstained_rate"))) },
    @{ Label = "Finance gate + verifier"; Value = Get-Percent $financeGateVerifier "answer_coverage"; Color = "#6172F3"; Tag = ("UNS " + (Format-Percent (Get-Percent $financeGateVerifier "unsupported_non_abstained_rate"))) },
    @{ Label = "Finance repair + verifier"; Value = Get-Percent $financeRepair "answer_coverage"; Color = "#12B76A"; Tag = ("UNS " + (Format-Percent (Get-Percent $financeRepair "unsupported_non_abstained_rate"))) }
)
Draw-HorizontalMetricChart "safety_vs_coverage_frontier.png" "Safety and Coverage on Fixed Splits" "Coverage is higher when the system answers more often; UNS is unsupported non-abstained rate." $safetyRows 100 "Answer coverage"

$asqaRepair3 = Get-RepairRow "asqa" "3b"
$financeRepair3 = Get-RepairRow "finance" "3b"
$repairRows = @(
    @{ Label = "Fixed examples"; Value = 300.0; Color = "#98A2B3"; Tag = "ASQA 200 + finance 100" },
    @{ Label = "Initial verifier pass"; Value = (Get-Number $asqaRepair3 "initial_verified_count") + (Get-Number $financeRepair3 "initial_verified_count"); Color = "#6172F3"; Tag = "accepted before repair" },
    @{ Label = "Repair attempted"; Value = (Get-Number $asqaRepair3 "repair_attempted_count") + (Get-Number $financeRepair3 "repair_attempted_count"); Color = "#F79009"; Tag = "format or evidence issue" },
    @{ Label = "Accepted after repair"; Value = (Get-Number $asqaRepair3 "accepted_after_repair_count") + (Get-Number $financeRepair3 "accepted_after_repair_count"); Color = "#12B76A"; Tag = "salvaged safely" },
    @{ Label = "Final abstentions"; Value = (Get-Number $asqaRepair3 "abstention_count") + (Get-Number $financeRepair3 "abstention_count"); Color = "#D0D5DD"; Tag = "insufficient support" }
)
Draw-HorizontalMetricChart "repair_funnel.png" "Repair-plus-Verifier Decision Funnel" "The verifier remains the final acceptance boundary; unsupported accepted repairs stayed at zero." $repairRows 300 "Examples"

$asqaDist = Get-DistractorRow "asqa"
$financeDist = Get-DistractorRow "finance"
$distractorRows = @(
    @{ Label = "ASQA coverage"; Value = Get-Percent $asqaDist "answer_coverage"; Color = "#12B76A" },
    @{ Label = "ASQA abstention"; Value = Get-Percent $asqaDist "abstention_rate"; Color = "#98A2B3" },
    @{ Label = "ASQA unsupported"; Value = Get-Percent $asqaDist "unsupported_non_abstained_rate"; Color = "#D92D20" },
    @{ Label = "Finance coverage"; Value = Get-Percent $financeDist "answer_coverage"; Color = "#12B76A" },
    @{ Label = "Finance abstention"; Value = Get-Percent $financeDist "abstention_rate"; Color = "#98A2B3" },
    @{ Label = "Finance unsupported"; Value = Get-Percent $financeDist "unsupported_non_abstained_rate"; Color = "#D92D20" }
)
Draw-HorizontalMetricChart "generated_distractor_robustness.png" "Generated Distractor Robustness" "The stress test injects misleading passages and checks whether unsafe citations are still rejected." $distractorRows 100 "Rate"

$unsRows = @(
    @{ Label = "ASQA baseline"; Value = Get-Percent $asqaBaseline "unsupported_non_abstained_rate"; Color = "#D92D20"; Tag = "fixed split" },
    @{ Label = "ASQA gate + verifier"; Value = Get-Percent $asqaGateVerifier "unsupported_non_abstained_rate"; Color = "#12B76A"; Tag = "fixed split" },
    @{ Label = "ASQA repair + verifier"; Value = Get-Percent $asqaRepair "unsupported_non_abstained_rate"; Color = "#12B76A"; Tag = "fixed split" },
    @{ Label = "Finance baseline"; Value = Get-Percent $financeBaseline "unsupported_non_abstained_rate"; Color = "#D92D20"; Tag = "fixed split" },
    @{ Label = "Finance gate + verifier"; Value = Get-Percent $financeGateVerifier "unsupported_non_abstained_rate"; Color = "#12B76A"; Tag = "fixed split" },
    @{ Label = "Finance repair + verifier"; Value = Get-Percent $financeRepair "unsupported_non_abstained_rate"; Color = "#12B76A"; Tag = "fixed split" }
)
Draw-HorizontalMetricChart "unsupported_non_abstained.png" "Unsupported Non-Abstained Answers" "Lower is better. The repair system preserves the verifier's zero-unsupported boundary." $unsRows 10 "Unsupported non-abstained rate"

$financeRows = @(
    @{ Label = "Baseline exact"; Value = Get-Percent $financeBaseline "exact_answer_accuracy"; Color = "#98A2B3"; Tag = "3B" },
    @{ Label = "Gate + verifier exact"; Value = Get-Percent $financeGateVerifier "exact_answer_accuracy"; Color = "#6172F3"; Tag = "3B" },
    @{ Label = "Repair + verifier exact"; Value = Get-Percent $financeRepair "exact_answer_accuracy"; Color = "#12B76A"; Tag = "3B" },
    @{ Label = "Baseline coverage"; Value = Get-Percent $financeBaseline "answer_coverage"; Color = "#D0D5DD"; Tag = "3B" },
    @{ Label = "Gate + verifier coverage"; Value = Get-Percent $financeGateVerifier "answer_coverage"; Color = "#BDB4FE"; Tag = "3B" },
    @{ Label = "Repair + verifier coverage"; Value = Get-Percent $financeRepair "answer_coverage"; Color = "#6CE9A6"; Tag = "3B" }
)
Draw-HorizontalMetricChart "finance_citation_accuracy.png" "Finance Accuracy and Answer Coverage" "The repair path improves exact-answer performance while preserving zero unsupported accepted answers." $financeRows 100 "Rate"

Write-Host "Rebuilt clean report figures in $AssetDir"
