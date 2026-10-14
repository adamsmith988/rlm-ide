/** Original single-weight line drawings in the spirit of the reference mockups. */

export function Circles({ count = 3, size = 520, className = "" }: { count?: number; size?: number; className?: string }) {
  return (
    <div className={`circles ${className}`}>
      <svg width="100%" height="100%" preserveAspectRatio="xMidYMid slice" viewBox={`0 0 ${size} ${size}`}>
        {Array.from({ length: count }, (_, i) => (
          <circle key={i} cx={size * (0.2 + i * 0.32)} cy={size * (0.25 + (i % 2) * 0.45)} r={size * (0.22 + i * 0.05)} fill="none" stroke="var(--circle)" strokeWidth={1.5} />
        ))}
      </svg>
    </div>
  );
}

/** A person sitting cross-legged with a laptop, thinking. */
export function PersonLaptop({ width = 260 }: { width?: number }) {
  return (
    <svg width={width} viewBox="0 0 260 240" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      {/* head + hair */}
      <circle cx="128" cy="52" r="20" />
      <path d="M109 46c3-15 16-22 29-18 9 3 13 10 12 18-6-6-14-8-22-7-7 1-13 4-19 7z" fill="currentColor" />
      <path d="M124 60c2 2 6 2 8 0" />
      {/* neck + torso */}
      <path d="M124 72v8M134 72v8" />
      <path d="M100 96c4-11 15-16 29-16s25 5 29 16l6 52H94z" />
      {/* arms to laptop */}
      <path d="M103 104c-8 16-10 30-4 40l30 2M155 104c8 16 10 30 4 40l-30 2" />
      {/* laptop */}
      <path d="M92 150h76l-10-36h-56z" fill="var(--bg)" />
      <path d="M84 150h92" />
      <circle cx="130" cy="131" r="3" />
      {/* legs crossed */}
      <path d="M94 150c-22 4-38 16-40 30 18 6 50 4 76-6M166 150c22 4 38 16 40 30-18 6-50 4-76-6" />
      <path d="M58 182c-6 2-10 6-8 9M202 182c6 2 10 6 8 9" />
      {/* thought marks */}
      <path d="M168 34c6-6 12-8 18-6M174 20c6-2 12 0 16 4" />
      {/* plant */}
      <path d="M28 220h36l-4-22H32z" />
      <path d="M46 198c-2-18-14-26-20-30M46 198c2-14 10-22 18-24M46 198c0-20 2-34 6-42" />
      <path d="M18 222h224" strokeWidth={1.5} />
    </svg>
  );
}

/** A person with headphones, reading a document, for empty agent chats. */
export function PersonHeadphones({ width = 160 }: { width?: number }) {
  return (
    <svg width={width} viewBox="0 0 160 160" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <circle cx="80" cy="54" r="18" />
      <path d="M62 54c0-14 8-24 18-24s18 10 18 24" />
      <rect x="56" y="50" width="8" height="14" rx="3" fill="currentColor" />
      <rect x="96" y="50" width="8" height="14" rx="3" fill="currentColor" />
      <path d="M74 60c3 3 9 3 12 0" />
      <path d="M58 132c0-28 10-44 22-44s22 16 22 44" />
      <path d="M54 104h52l-6 34H60z" fill="var(--bg)" />
      <path d="M64 114h32M64 122h24" strokeWidth={1.5} />
      <circle cx="80" cy="80" r="66" stroke="var(--circle)" strokeWidth={1.5} />
    </svg>
  );
}

/** Small logo mark: an open circle with a REPL caret. */
export function LogoMark({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.2} strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <circle cx="12" cy="12" r="9.5" />
      <path d="m8.5 9 3 3-3 3M13 15h3" />
    </svg>
  );
}
