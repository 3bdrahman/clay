// ChartRenderer — Recharts-based chart for data analysis results

import {
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import type { ChartConfig } from '../lib/types';

const COLORS = ['#1184e7', '#0e3a6f', '#3a9ff8', '#7fc1fc', '#bbdcfd', '#0567c4', '#0752a0', '#0a4585'];

interface ChartRendererProps {
  config: ChartConfig;
}

export default function ChartRenderer({ config }: ChartRendererProps) {
  const type = config.type;
  const title = config.title;
  const xKey = config.xKey;
  const yKeys = config.yKeys;
  const data = config.data;

  if (type === 'bar') {
    return (
      <div>
        <div className="text-[11px] font-semibold text-ink-700 dark:text-ink-300 mb-2" role="heading" aria-level={3}>
          {title}
        </div>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 24 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.15} />
            <XAxis
              dataKey={xKey}
              tick={{ fontSize: 10, fill: 'currentColor' }}
              angle={data.length > 4 ? -25 : 0}
              textAnchor={data.length > 4 ? 'end' : 'middle'}
              height={data.length > 4 ? 50 : 30}
            />
            <YAxis tick={{ fontSize: 10, fill: 'currentColor' }} />
            <Tooltip
              contentStyle={{
                fontSize: 11,
                background: 'rgba(255,255,255,0.95)',
                border: '1px solid #dadee5',
                borderRadius: 4,
              }}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            {yKeys.map((y, i) => (
              <Bar key={y} dataKey={y} fill={COLORS[i % COLORS.length]} radius={[4, 4, 0, 0]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  }

  if (type === 'line') {
    return (
      <div>
        <div className="text-[11px] font-semibold text-ink-700 dark:text-ink-300 mb-2" role="heading" aria-level={3}>
          {title}
        </div>
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 24 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.15} />
            <XAxis dataKey={xKey} tick={{ fontSize: 10, fill: 'currentColor' }} />
            <YAxis tick={{ fontSize: 10, fill: 'currentColor' }} />
            <Tooltip />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            {yKeys.map((y, i) => (
              <Line
                key={y}
                type="monotone"
                dataKey={y}
                stroke={COLORS[i % COLORS.length]}
                strokeWidth={2}
                dot={{ r: 3 }}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    );
  }

  if (type === 'pie') {
    return (
      <div>
        <div className="text-[11px] font-semibold text-ink-700 dark:text-ink-300 mb-2" role="heading" aria-level={3}>
          {title}
        </div>
        <ResponsiveContainer width="100%" height={240}>
          <PieChart>
            <Pie
              data={data}
              dataKey={yKeys[0]}
              nameKey={xKey}
              cx="50%"
              cy="50%"
              outerRadius={80}
              label={(d: unknown) => String((d as Record<string, unknown>)[xKey])}
            >
              {data.map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend wrapperStyle={{ fontSize: 11 }} />
          </PieChart>
        </ResponsiveContainer>
      </div>
    );
  }

  return null;
}
