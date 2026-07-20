<script lang="ts">
  import { tabStore, setTab, run, hwStore, themeStore, toggleTheme, uiMode } from '../store.svelte'
  import type { Tab } from '../store.svelte'
  import { t } from '../i18n.svelte'

  const simpleTabs: { id: Tab; key: string; iconClass: string }[] = [
    { id: 'train',  key: 'nav.teach',   iconClass: 'bi-fire' },
    { id: 'merge',  key: 'nav.combine', iconClass: 'bi-lightning-charge' },
    { id: 'runs',   key: 'nav.history', iconClass: 'bi-clock-history' },
    { id: 'settings', key: 'nav.settings', iconClass: 'bi-gear' },
    { id: 'help',   key: 'nav.help',    iconClass: 'bi-question-circle' },
  ]
  const advancedTabs: { id: Tab; key: string; iconClass: string }[] = [
    { id: 'train',  key: 'nav.train',  iconClass: 'bi-fire' },
    { id: 'merge',  key: 'nav.merge',  iconClass: 'bi-lightning-charge' },
    { id: 'runs',   key: 'nav.runs',   iconClass: 'bi-clock-history' },
    { id: 'guider', key: 'nav.guider', iconClass: 'bi-compass' },
    { id: 'settings', key: 'nav.settings', iconClass: 'bi-gear' },
    { id: 'help',   key: 'nav.help',   iconClass: 'bi-question-circle' },
  ]
  const tabs = $derived(uiMode.advanced ? advancedTabs : simpleTabs)
</script>

<aside class="sidebar">
  <!-- Logo -->
  <div class="logo">
    <div class="logo-mark" style="background: transparent; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; border-radius: 0; padding: 0;">
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="116 256 836 510" style="width: 100%; height: 100%; fill: var(--color-brand)">
        <path d="M 936 256 L 935 257 L 844 257 L 843 258 L 840 258 L 839 259 L 838 259 L 837 260 L 835 260 L 834 261 L 833 261 L 832 262 L 831 262 L 829 264 L 828 264 L 826 266 L 825 266 L 812 279 L 812 280 L 793 299 L 793 300 L 773 320 L 773 321 L 752 342 L 752 343 L 733 362 L 733 363 L 714 382 L 714 383 L 696 401 L 696 402 L 679 419 L 679 420 L 662 437 L 662 438 L 647 453 L 647 454 L 645 456 L 621 432 L 621 431 L 594 404 L 594 403 L 577 386 L 489 386 L 489 387 L 507 405 L 507 406 L 527 426 L 527 427 L 547 447 L 547 448 L 567 468 L 567 469 L 589 491 L 589 492 L 600 503 L 600 504 L 604 508 L 604 509 L 606 511 L 606 512 L 607 513 L 607 514 L 608 515 L 608 516 L 609 517 L 609 518 L 610 519 L 610 520 L 611 521 L 611 522 L 612 523 L 612 526 L 613 527 L 613 537 L 614 538 L 614 754 L 615 755 L 615 766 L 619 762 L 620 762 L 636 746 L 637 746 L 654 729 L 655 729 L 673 711 L 674 711 L 689 696 L 689 695 L 692 692 L 692 691 L 694 689 L 694 688 L 695 687 L 695 686 L 696 685 L 696 680 L 697 679 L 697 559 L 698 558 L 698 520 L 699 519 L 699 518 L 703 514 L 703 513 L 727 489 L 727 488 L 760 455 L 760 454 L 795 419 L 795 418 L 830 383 L 830 382 L 863 349 L 863 348 L 896 315 L 896 314 L 938 272 L 938 271 L 952 257 L 952 256 Z M 568 256 L 567 257 L 253 257 L 252 258 L 246 258 L 245 259 L 241 259 L 240 260 L 237 260 L 236 261 L 232 261 L 231 262 L 229 262 L 228 263 L 226 263 L 225 264 L 223 264 L 222 265 L 221 265 L 220 266 L 219 266 L 218 267 L 216 267 L 215 268 L 214 268 L 213 269 L 212 269 L 211 270 L 210 270 L 209 271 L 208 271 L 207 272 L 206 272 L 205 273 L 204 273 L 202 275 L 201 275 L 199 277 L 198 277 L 196 279 L 195 279 L 193 281 L 192 281 L 188 285 L 187 285 L 180 292 L 179 292 L 178 293 L 178 294 L 170 302 L 170 303 L 167 306 L 167 307 L 164 310 L 164 311 L 162 313 L 162 314 L 161 315 L 161 316 L 159 318 L 159 319 L 158 320 L 158 321 L 157 322 L 157 323 L 156 324 L 156 325 L 155 326 L 155 327 L 154 328 L 154 330 L 153 331 L 153 332 L 152 333 L 152 335 L 151 336 L 151 338 L 150 339 L 150 341 L 149 342 L 149 344 L 148 345 L 148 348 L 147 349 L 147 352 L 146 353 L 146 360 L 145 361 L 145 389 L 146 390 L 146 396 L 147 397 L 147 400 L 148 401 L 148 404 L 149 405 L 149 408 L 150 409 L 150 411 L 151 412 L 151 414 L 152 415 L 152 417 L 153 418 L 153 419 L 154 420 L 154 422 L 155 423 L 155 424 L 156 425 L 156 426 L 157 427 L 157 428 L 158 429 L 158 430 L 160 432 L 160 433 L 161 434 L 161 435 L 163 437 L 163 438 L 165 440 L 165 441 L 169 445 L 169 446 L 175 452 L 175 453 L 181 459 L 182 459 L 188 465 L 189 465 L 192 468 L 193 468 L 195 470 L 196 470 L 198 472 L 199 472 L 201 474 L 202 474 L 203 475 L 204 475 L 206 477 L 208 477 L 209 478 L 210 478 L 211 479 L 212 479 L 213 480 L 214 480 L 215 481 L 216 481 L 217 482 L 219 482 L 220 483 L 222 483 L 223 484 L 225 484 L 226 485 L 228 485 L 229 486 L 231 486 L 232 487 L 235 487 L 236 488 L 239 488 L 240 489 L 247 489 L 248 490 L 331 490 L 332 491 L 427 491 L 428 492 L 430 492 L 431 493 L 432 493 L 433 494 L 435 494 L 436 495 L 437 495 L 439 497 L 440 497 L 481 538 L 481 539 L 491 549 L 491 550 L 493 552 L 493 553 L 494 554 L 494 555 L 495 556 L 495 568 L 494 569 L 494 571 L 493 572 L 493 574 L 492 575 L 492 576 L 491 577 L 491 578 L 490 579 L 490 580 L 480 590 L 479 590 L 478 591 L 477 591 L 476 592 L 475 592 L 474 593 L 473 593 L 472 594 L 471 594 L 470 595 L 468 595 L 467 596 L 207 596 L 116 687 L 462 687 L 463 686 L 475 686 L 476 685 L 480 685 L 481 684 L 484 684 L 485 683 L 487 683 L 488 682 L 490 682 L 491 681 L 494 681 L 495 680 L 496 680 L 497 679 L 499 679 L 500 678 L 501 678 L 502 677 L 504 677 L 505 676 L 506 676 L 507 675 L 508 675 L 509 674 L 510 674 L 511 673 L 512 673 L 514 671 L 515 671 L 516 670 L 517 670 L 519 668 L 520 668 L 521 667 L 522 667 L 524 665 L 525 665 L 529 661 L 530 661 L 534 657 L 535 657 L 548 644 L 548 643 L 552 639 L 552 638 L 556 634 L 556 633 L 558 631 L 558 630 L 560 628 L 560 627 L 561 626 L 561 625 L 563 623 L 563 622 L 564 621 L 564 620 L 565 619 L 565 618 L 566 617 L 566 616 L 567 615 L 567 614 L 568 613 L 568 612 L 569 611 L 569 610 L 570 609 L 570 607 L 571 606 L 571 605 L 572 604 L 572 602 L 573 601 L 573 599 L 574 598 L 574 596 L 575 595 L 575 592 L 576 591 L 576 588 L 577 587 L 577 582 L 578 581 L 578 575 L 579 574 L 579 551 L 578 550 L 578 545 L 577 544 L 577 541 L 576 540 L 576 538 L 575 537 L 575 535 L 574 534 L 574 533 L 573 532 L 573 531 L 572 530 L 572 529 L 571 528 L 571 527 L 568 524 L 568 523 L 559 514 L 559 513 L 538 492 L 538 491 L 509 462 L 509 461 L 475 427 L 474 427 L 471 424 L 470 424 L 468 422 L 467 422 L 465 420 L 464 420 L 463 419 L 462 419 L 461 418 L 460 418 L 459 417 L 458 417 L 457 416 L 456 416 L 455 415 L 453 415 L 452 414 L 451 414 L 450 413 L 448 413 L 447 412 L 445 412 L 444 411 L 441 411 L 440 410 L 438 410 L 437 409 L 434 409 L 433 408 L 421 408 L 420 407 L 285 407 L 284 406 L 249 406 L 249 405 L 249 405 L 249 404 L 249 403 L 249 401 L 249 394 L 249 393 L 249 391 L 249 390 L 249 389 L 249 388 L 249 387 L 249 386 L 249 385 L 249 383 L 249 382 L 249 369 L 249 368 L 249 365 L 249 364 L 249 363 L 249 362 L 249 361 L 249 359 L 249 358 L 249 355 L 249 354 L 249 352 L 249 351 L 249 348 L 249 347 L 249 346 L 249 345 L 250 344 L 411 344 L 412 343 L 546 343 L 584 305 L 584 304 L 631 257 L 631 256 Z" />
      </svg>
    </div>
    <div class="logo-text">
      <span class="logo-name">Sytra</span>
      <span class="logo-sub">STUDIO</span>
    </div>
  </div>

  <!-- Nav -->
  <nav class="nav" aria-label="Main navigation">
    {#each tabs as tab}
      <button
        id="tab-{tab.id}"
        class="nav-item"
        class:active={tabStore.active === tab.id}
        onclick={() => setTab(tab.id)}
        aria-current={tabStore.active === tab.id ? 'page' : undefined}
      >
        <span class="nav-icon" style="display: flex; align-items: center; justify-content: center; width: 16px; height: 16px; font-size: 14px">
          <i class="bi {tab.iconClass}"></i>
        </span>
        <span class="nav-label">{t(tab.key)}</span>
        {#if tab.id === run.kind && run.status === 'running'}
          <span class="pulse-dot" style="width:6px;height:6px;margin-left:auto"></span>
        {/if}
      </button>
    {/each}
  </nav>

  <div style="flex:1"></div>

  <!-- HW info -->
  <div class="hw-info">
    {#if hwStore.info}
      <div class="hw-title">{t('sidebar.hardware')}</div>
      <div class="hw-row">
        <span class="hw-label">{t('sidebar.backend')}</span>
        <span class="badge badge-info" style="font-size:10px">{hwStore.info.backend.toUpperCase()}</span>
      </div>
      <div class="hw-row">
        <span class="hw-label">VRAM</span>
        <span class="hw-val">{(hwStore.info.vram_mb / 1024).toFixed(0)} GB</span>
      </div>
      <div class="hw-row">
        <span class="hw-label">RAM</span>
        <span class="hw-val">{(hwStore.info.ram_mb / 1024).toFixed(0)} GB</span>
      </div>
    {:else}
      <span class="text-small">{t('sidebar.detecting')}</span>
    {/if}
  </div>

  <!-- Bottom actions -->
  <div class="sidebar-bottom">
    <button
      class="btn btn-ghost btn-icon theme-toggle"
      onclick={toggleTheme}
      id="btn-theme-toggle"
      data-tooltip={themeStore.dark ? 'Switch to light mode' : 'Switch to dark mode'}
      data-tooltip-align="left"
      aria-label="Toggle dark mode"
      style="display: flex; align-items: center; justify-content: center; width: 32px; height: 32px; font-size: 14px"
    >
      <i class="bi {themeStore.dark ? 'bi-sun-fill' : 'bi-moon-fill'}"></i>
    </button>
    <span class="version">v1.0.0</span>
  </div>
</aside>

<style>
  .sidebar {
    width: var(--sidebar-width);
    background: var(--color-surface);
    border-right: 1px solid var(--color-border);
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
    overflow: hidden;
    padding: var(--space-3) var(--space-2);
    gap: var(--space-2);
  }

  /* Logo */
  .logo {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-2) var(--space-2) var(--space-3);
    border-bottom: 1px solid var(--color-border);
    margin-bottom: var(--space-1);
  }
  .logo-mark {
    width: 28px; height: 28px;
    background: transparent;
    border-radius: 0;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
  }
  .logo-text { display: flex; flex-direction: column; line-height: 1.1; }
  .logo-name {
    font-family: var(--font-display);
    font-weight: 700;
    font-size: 17px;
    letter-spacing: -0.02em;
    color: var(--color-ink);
  }
  .logo-sub  { font-size: 9px; color: var(--color-ink-ghost); letter-spacing: 0.22em; }

  /* Nav */
  .nav { display: flex; flex-direction: column; gap: 1px; }
  .nav-item {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    padding: 8px var(--space-3);
    border-radius: 0;
    border: none;
    border-left: 2px solid transparent;
    background: transparent;
    color: var(--color-ink-subtle);
    font-family: var(--font-sans);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    cursor: pointer;
    text-align: left;
    transition: all var(--dur-fast) var(--ease);
    width: 100%;
  }
  .nav-item:hover  { background: var(--color-surface-muted); color: var(--color-ink); }
  .nav-item.active {
    background: var(--color-brand-subtle);
    border-left-color: var(--color-brand);
    color: var(--color-brand);
    font-weight: 600;
  }
  .nav-icon { font-size: 14px; width: 18px; text-align: center; flex-shrink: 0; }
  .nav-label { flex: 1; }

  /* HW */
  .hw-info {
    background: var(--color-surface-muted);
    border-radius: var(--radius-sm);
    padding: var(--space-3);
    display: flex;
    flex-direction: column;
    gap: 4px;
    border: 1px solid var(--color-border);
  }
  .hw-title { font-size: 10px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--color-ink-ghost); margin-bottom: 2px; }
  .hw-row   { display: flex; align-items: center; justify-content: space-between; }
  .hw-label { font-size: 11px; color: var(--color-ink-ghost); }
  .hw-val   { font-size: 11px; font-weight: 500; font-family: var(--font-mono); color: var(--color-ink-subtle); }

  /* Bottom */
  .sidebar-bottom {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--space-2) var(--space-1);
    border-top: 1px solid var(--color-border);
    margin-top: var(--space-1);
  }
  .theme-toggle { font-size: 16px; color: var(--color-ink-ghost); }
  .theme-toggle:hover { color: var(--color-ink); }
  .version { font-size: 11px; color: var(--color-ink-ghost); }
</style>
