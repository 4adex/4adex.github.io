'use client';

import { useEffect, useState } from 'react';

interface ContributionDay {
    date: string;
    count: number;
    level: 0 | 1 | 2 | 3 | 4;
}

interface GitHubGraphProps {
    username: string;
}

interface APIContribution {
    date: string;
    count: number;
    level: number;
}

interface APIResponse {
    total: {
        [year: string]: number;
    };
    contributions: APIContribution[];
}

function getMonthLabels(weeks: ContributionDay[][]): { label: string; index: number }[] {
    const months: { label: string; index: number }[] = [];
    const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    let lastMonth = -1;

    weeks.forEach((week, weekIndex) => {
        // Use the first day of the week to determine the month
        const firstDayOfWeek = week[0];
        if (!firstDayOfWeek) return;

        const date = new Date(firstDayOfWeek.date);
        const month = date.getMonth();

        if (month !== lastMonth) {
            // Skip if this is the first week and it's a partial week from previous month
            // (i.e., the month changes within the first week)
            if (weekIndex === 0) {
                // Check if most days in first week belong to this month
                const daysInThisMonth = week.filter(d => new Date(d.date).getMonth() === month).length;
                if (daysInThisMonth < 4) {
                    lastMonth = month;
                    return;
                }
            }
            months.push({ label: monthNames[month], index: weekIndex });
            lastMonth = month;
        }
    });

    return months;
}

export default function GitHubGraph({ username }: GitHubGraphProps) {
    const [data, setData] = useState<ContributionDay[]>([]);
    const [totalContributions, setTotalContributions] = useState(0);
    const [hoveredDay, setHoveredDay] = useState<ContributionDay | null>(null);
    const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        async function fetchContributions() {
            try {
                setLoading(true);
                setError(null);

                const response = await fetch(
                    `https://github-contributions-api.jogruber.de/v4/${username}?y=last`
                );

                if (!response.ok) {
                    throw new Error('Failed to fetch GitHub contributions');
                }

                const result: APIResponse = await response.json();

                const contributions: ContributionDay[] = result.contributions.map((c) => ({
                    date: c.date,
                    count: c.count,
                    level: Math.min(c.level, 4) as 0 | 1 | 2 | 3 | 4,
                }));

                contributions.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());

                const firstDate = new Date(contributions[0]?.date);
                const startDay = firstDate.getDay();
                const paddedContributions: ContributionDay[] = [];

                for (let i = 0; i < startDay; i++) {
                    const padDate = new Date(firstDate);
                    padDate.setDate(padDate.getDate() - (startDay - i));
                    paddedContributions.push({
                        date: padDate.toISOString().split('T')[0],
                        count: 0,
                        level: 0,
                    });
                }

                paddedContributions.push(...contributions);

                setData(paddedContributions);

                const total = Object.values(result.total).reduce((sum, val) => sum + val, 0);
                setTotalContributions(total);
            } catch (err) {
                setError(err instanceof Error ? err.message : 'Failed to load contributions');
                console.error('GitHub contributions error:', err);
            } finally {
                setLoading(false);
            }
        }

        if (username) {
            fetchContributions();
        }
    }, [username]);

    const handleMouseEnter = (day: ContributionDay, event: React.MouseEvent) => {
        const rect = event.currentTarget.getBoundingClientRect();
        setHoveredDay(day);
        setTooltipPosition({
            x: rect.left + rect.width / 2,
            y: rect.top - 10,
        });
    };

    const handleMouseLeave = () => {
        setHoveredDay(null);
    };

    const weeks: ContributionDay[][] = [];
    for (let i = 0; i < data.length; i += 7) {
        weeks.push(data.slice(i, i + 7));
    }

    const monthLabels = getMonthLabels(weeks);

    const formatDate = (dateStr: string) => {
        const date = new Date(dateStr);
        return date.toLocaleDateString('en-US', {
            weekday: 'short',
            month: 'short',
            day: 'numeric',
            year: 'numeric',
        });
    };

    if (loading) {
        return <div className="github-graph-loading">Loading contributions...</div>;
    }

    if (error) {
        return (
            <div className="github-graph-container">
                <div className="github-graph-error">
                    Unable to load GitHub contributions for @{username}
                </div>
            </div>
        );
    }

    if (data.length === 0) {
        return null;
    }

    return (
        <div className="github-graph-container">
            <div className="github-graph-header">
                <h3>{totalContributions.toLocaleString()} contributions in the last year</h3>
                <a
                    href={`https://github.com/${username}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="github-profile-link"
                >
                    @{username}
                </a>
            </div>

            <div className="github-graph-wrapper">
                <div className="github-graph-months">
                    {monthLabels.map((month, i) => (
                        <span
                            key={i}
                            className="github-graph-month"
                            style={{ gridColumn: month.index + 1 }}
                        >
                            {month.label}
                        </span>
                    ))}
                </div>

                <div className="github-graph-main">
                    <div className="github-graph-grid">
                        {weeks.map((week, weekIndex) => (
                            <div key={weekIndex} className="github-graph-week">
                                {week.map((day, dayIndex) => (
                                    <div
                                        key={`${weekIndex}-${dayIndex}`}
                                        className={`github-graph-cell level-${day.level}`}
                                        onMouseEnter={(e) => handleMouseEnter(day, e)}
                                        onMouseLeave={handleMouseLeave}
                                    />
                                ))}
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            <div className="github-graph-legend">
                <span>Less</span>
                <div className="github-graph-cell level-0" />
                <div className="github-graph-cell level-1" />
                <div className="github-graph-cell level-2" />
                <div className="github-graph-cell level-3" />
                <div className="github-graph-cell level-4" />
                <span>More</span>
            </div>

            {hoveredDay && (
                <div
                    className="github-graph-tooltip"
                    style={{
                        position: 'fixed',
                        left: tooltipPosition.x,
                        top: tooltipPosition.y,
                        transform: 'translate(-50%, -100%)',
                    }}
                >
                    <strong>
                        {hoveredDay.count} contribution{hoveredDay.count !== 1 ? 's' : ''}
                    </strong>
                    <span>{formatDate(hoveredDay.date)}</span>
                </div>
            )}
        </div>
    );
}
