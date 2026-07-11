# Moves the Windows pagefile off the C: SSD onto the D: HDD.
# C: keeps a small 2 GB pagefile (kernel crash dumps); D: gets 16-32 GB.
# Writes the registry value directly (authoritative). Effective after reboot.

$ErrorActionPreference = 'Stop'

$key = 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management'
$value = @('c:\pagefile.sys 2048 2048', 'd:\pagefile.sys 16384 32768')
Set-ItemProperty -Path $key -Name PagingFiles -Value $value -Type MultiString

Write-Host "PagingFiles set to:"
(Get-ItemProperty $key).PagingFiles | ForEach-Object { Write-Host "  $_" }
Write-Host ""
Write-Host "Done. Takes effect after the next REBOOT."
Write-Host "(Wait for the running merge to finish before rebooting.)"
Start-Sleep -Seconds 8
