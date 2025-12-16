'use client';

import { useState, FormEvent } from 'react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Send, Dices, Loader2 } from 'lucide-react';

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  disabled?: boolean;
}

// Pre-configured test messages for testing CogCanvas extraction features
const TEST_MESSAGES = [
  // Short messages
  "我们需要在下周五之前完成用户认证模块的开发。",
  "决定采用 PostgreSQL 作为主数据库。",
  "请记得明天下午3点参加项目评审会议。",

  // Medium messages
  "关于新功能的技术选型，我倾向于使用 React + TypeScript。主要原因是：第一，团队成员都比较熟悉这个技术栈；第二，生态系统成熟，有丰富的组件库；第三，TypeScript 可以提高代码质量。大家觉得怎么样？",

  "今天的待办事项：1. 完成用户登录页面的UI设计 2. 修复搜索功能的bug 3. 更新API文档 4. 参加下午的站会 5. Code review张三提交的PR。",

  "我发现现在的数据加载速度比较慢，可能是因为没有做分页。建议我们添加分页功能，每页显示20条数据，并且加上懒加载优化用户体验。",

  // Long messages with lists and multiple paragraphs
  `项目进度汇报：

本周完成的工作：
- 完成了前端框架搭建，选用了 Next.js 14 和 Tailwind CSS
- 实现了用户认证模块，包括登录、注册和密码重置
- 完成了基础的数据模型设计
- 编写了单元测试，测试覆盖率达到 75%

下周计划：
1. 开发核心业务功能模块
2. 集成第三方支付接口
3. 完成响应式设计适配
4. 进行性能优化和安全测试

遇到的问题：
- API响应时间偏长，需要优化数据库查询
- 部分UI组件在移动端显示异常，需要调整样式`,

  `技术决策：关于微服务架构的讨论

背景：
随着业务增长，现有的单体应用越来越难以维护。我们需要考虑是否要迁移到微服务架构。

优点分析：
- 服务独立部署，可以单独扩展
- 技术栈灵活，不同服务可以用不同技术
- 故障隔离，一个服务挂了不影响其他服务
- 团队可以并行开发，提高效率

缺点分析：
- 系统复杂度增加，需要处理分布式事务
- 运维成本上升，需要更多的监控和日志系统
- 网络延迟可能影响性能
- 团队需要学习新的技术和工具

我的建议：
暂时不做完全的微服务改造，可以先采用模块化单体架构，为未来的微服务化做准备。等团队规模扩大、业务更加成熟后再逐步拆分。`,

  `会议纪要 - 产品需求评审会议

时间：2024年1月15日 14:00-16:00
参与人员：产品经理王芳、前端负责人李明、后端负责人张伟、UI设计师刘洋

讨论内容：

1. 新版本主要功能
   - 实时协作编辑功能（优先级：高）
   - 数据可视化图表（优先级：中）
   - 导出PDF功能（优先级：低）
   - 权限管理优化（优先级：高）

2. 技术可行性评估
   李明：实时协作可以用 WebSocket 实现，预计需要2周开发时间
   张伟：后端需要重构权限系统，可能需要3周
   刘洋：UI设计稿下周可以交付

3. 时间规划
   - 第一周：需求细化和技术方案设计
   - 第二周：开始开发实时协作功能
   - 第三-四周：完成权限系统重构
   - 第五周：集成测试和bug修复

待办事项：
- 王芳负责整理详细需求文档（截止日期：1月20日）
- 李明负责WebSocket技术方案（截止日期：1月22日）
- 张伟负责权限系统设计文档（截止日期：1月22日）
- 刘洋负责完成UI设计稿（截止日期：1月22日）

下次会议：1月25日 14:00`,

  `问题分析：用户反馈登录失败率较高

问题描述：
过去一周收到15起用户反馈登录失败的问题，失败率从之前的2%上升到8%。

排查过程：
1. 检查了服务器日志，发现大量的认证超时错误
2. 分析了数据库性能，查询响应时间正常
3. 检查了网络状况，发现CDN节点有部分异常
4. 查看了前端错误日志，发现有跨域请求被拦截

根本原因：
上周更新了nginx配置，CORS设置有误，导致部分请求被拦截。同时，会话token的过期时间设置太短（5分钟），用户体验不好。

解决方案：
1. 立即修复nginx的CORS配置（紧急）
2. 将token过期时间延长到30分钟（优先级高）
3. 添加token自动刷新机制（优先级中）
4. 增加前端错误提示，让用户知道具体失败原因（优先级中）

预期效果：
修复后登录失败率应该降低到2%以下，用户体验明显改善。`,

  "我们是继续使用 REST API 还是切换到 GraphQL？各有什么优劣？",

  "记录一个重要事实：系统在高峰期（每天上午10-11点）的并发用户数可以达到5000人，目前服务器配置是4核8G，CPU使用率经常超过80%。",

  "下个迭代的核心目标是提升用户留存率，我们需要添加个性化推荐功能、优化首页加载速度、增加用户引导流程。",
];

function getRandomTestMessage(): string {
  const randomIndex = Math.floor(Math.random() * TEST_MESSAGES.length);
  return TEST_MESSAGES[randomIndex];
}

export function ChatInput({ onSendMessage, disabled }: ChatInputProps) {
  const [input, setInput] = useState('');

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (input.trim() && !disabled) {
      onSendMessage(input.trim());
      setInput('');
    }
  };

  const handleTestMessage = () => {
    if (!disabled) {
      const testMessage = getRandomTestMessage();
      onSendMessage(testMessage);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex gap-2 p-4 border-t">
      <Input
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="Type your message..."
        disabled={disabled}
        className="flex-1"
      />
      <Button
        type="button"
        variant="outline"
        onClick={handleTestMessage}
        disabled={disabled}
        title="Send random test message"
      >
        <Dices className="h-4 w-4" />
      </Button>
      <Button type="submit" disabled={disabled || !input.trim()}>
        {disabled ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          <Send className="h-4 w-4" />
        )}
      </Button>
    </form>
  );
}
