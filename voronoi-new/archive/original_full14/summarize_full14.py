"""Finalize the full-video audit and a human-readable delivery note."""
from pathlib import Path
from collections import Counter
import json,shutil

HERE=Path(__file__).resolve().parent
OUT=HERE/'full-14-10.20'


def main():
    validation=json.loads((OUT/'validation.json').read_text(encoding='utf-8'))
    rows=[json.loads(line) for line in (OUT/'frames.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['frame'] for r in rows]==list(range(validation['frames']))
    assert all(r['vertices_per_mask']==64 and r['required_polyline_missing_pixels']==0 for r in rows)
    issue_frames={}
    examples={}
    intersections=Counter()
    for row in rows:
        for chord in row['chords']:
            intersections[chord['intersection_count']]+=1
        for issue in row['unresolved']:
            status=issue['status']
            issue_frames.setdefault(status,set()).add(row['frame'])
            examples.setdefault(status,dict(frame=row['frame'],seconds=row['seconds'],issue=issue))
    audit=dict(sequential_frame_records=True,all_frames_keep_64_vertices=True,
               required_polyline_pixels_present_every_frame=True,
               polyline_intersection_counts=dict(sorted(intersections.items())),
               issue_frame_counts={key:len(value) for key,value in issue_frames.items()},
               issue_frame_indices={key:sorted(value) for key,value in issue_frames.items()},
               first_issue_examples=examples,
               frames_with_unresolved_overlap=sum(bool(r['unresolved']) for r in rows),
               frames_with_failed_junctions=sum(r['failed_junctions']>0 for r in rows),
               mean_failed_junctions=sum(r['failed_junctions'] for r in rows)/len(rows))
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    snap=OUT/'scripts_snapshot';snap.mkdir(exist_ok=True)
    for name in ('render_full14.py','prepare_full14.py','topology_voronoi.py','bvh_field.py',
                 'overlap_chords.py','fast_voronoi.py','baseline_video.py',
                 'check_full14_geometry.py','summarize_full14.py'):
        shutil.copy2(HERE/name,snap/name)
    names={'contour_order_disagrees':'两个轮廓给出的交点顺序不一致',
           'triple_overlap':'三个或更多掩码重叠',
           'needs_rule_for_crossings_or_containment':'包含或不足两个可连接交点',
           'ambiguous_side_assignment':'分割后的区域归属不明确',
           'self_crossing_interface':'连接交点会导致分界线自交',
           'invalid_polygon':'几何修复后仍无有效多边形'}
    issue_lines='\n'.join(f'- {names.get(k,k)}：{len(v)} 帧。' for k,v in issue_frames.items())
    text=f'''完整结果：14-10.20_voronoi64_polyline.mp4

输入为用户指定的 14-10.20.mp4，保留其掩码、编号、置信度标注，新增白色维诺网络。输出 2510 × 676、30 fps、2901 帧、96.7 秒；已逐帧解码检查帧数、尺寸和帧率。原视频没有音轨。

重叠分界使用已确认的规则：沿轮廓顺序列出全部交点，再连接 1–2、2–3、3–4……形成开放连续折线，不连接末点回首点。排序以轮廓弧长为依据，在最大交点间隔处断开，并检查两个轮廓的排序一致性。

每个掩码保留 64 个原始轮廓点，距离计算使用它们组成的完整 64 条线段。先按原视频的跟踪清理、插值和平滑设置重建轮廓；80 条轨迹的编号、类别、出现范围和检测帧数与原记录匹配。部分平滑轮廓有细小自交，仅在布尔运算中用 make_valid 修复拓扑，不改变原始 64 点或距离计算线段。6 个抽查帧的 47000 次点内外查询、交点归属、分区面积守恒以及轮廓未修改检查均通过。

掩码外的网络采用线段 BVH 最近邻查询、网格边上完整多边形距离求根、共享边交点与三角网格连接；掩码内部的普通维诺线会被清除，然后单独加入约定的重叠折线。网格间距为 2 像素，求根后的线条按原图分辨率绘制，线宽 2 像素。这里的网格是计算空间的采样，不是把掩码轮廓缩减成少数点。

运行使用 CPU 上的 3 个帧处理进程，每个 BVH 查询使用 4 个线程；输出 H.264，CRF 18。全片渲染和编码耗时 {validation['render_seconds']:.1f} 秒，端到端吞吐 {validation['frames']/validation['render_seconds']:.2f} 帧/秒（不含预处理和末尾解码验证）。轮廓预处理耗时见 preparation.json。该耗时不应直接与不同网格配置、不同绘制逻辑的旧基线宣称等价加速倍数。

生成了 {sum(intersections.values())} 条“掩码对 × 帧”重叠折线，交点数量分布为 {dict(sorted(intersections.items()))}。每帧均检查规定折线的像素在最终未压缩网络中完整保留。

仍有 {audit['frames_with_unresolved_overlap']} 帧涉及无法由当前规则唯一处理的重叠，相关位置没有强行添加分隔线；不影响该帧其他正常掩码的绘制。分类帧数可重叠：
{issue_lines}

另外，外部网络仍是网格数值重建，并非解析的精确线段维诺图。遇到未收敛的三岔点或无法括根的网格边会略去局部线段，可能留下细小缺口；计数保存在 validation.json 和逐帧日志中。因此本版完成了整段视频处理，但不能声称所有局部拓扑问题都已消除。

文件索引：
- contact_sheet.jpg：6 帧总览。
- overlap_detail_f2400.png、overlap_detail_f2850.png：带交点编号的局部放大，仅用于核对，视频中没有这些额外编号。
- validation.json：视频解码验证、耗时、汇总与脚本哈希。
- audit.json：所有异常帧索引及首次异常示例，帧号从 0 开始。
- frames.jsonl：逐帧交点坐标、轨迹编号、耗时、几何修复与异常记录。
- geometry_validation.json：原轮廓保留及几何修复检查。
- frames64.pkl：与视频对齐的 64 点轮廓缓存。
- scripts_snapshot：此次使用的脚本快照；运行入口为上一级 render_full14.py，依赖在上一级 deps。

复现命令（在 E:\\AAA-reseach\\videos\\new-track 中）：
& 'E:\\AAA-reseach\\env\\yolo26\\python.exe' -u 'voronoi-test\\render_full14.py' --workers 3
'''
    (OUT/'README.md').write_text(text,encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
