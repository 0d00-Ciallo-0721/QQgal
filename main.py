import json
import datetime
from typing import Dict, List, Optional, Set, Any

from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
import astrbot.api.message_components as Comp
from astrbot.core.utils.session_waiter import session_waiter, SessionController
from astrbot.api import AstrBotConfig, logger  # 修改为导入astrbot的logger

@register("astrbot_plugin_QQgal", "和泉智宏", "Galgame", "1.2", "https://github.com/0d00-Ciallo-0721/astrbot_plugin_QQgal")
class GalGamePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        """
        初始化 GalGame 插件。
        :param context: AstrBot 插件上下文
        :param config: 插件配置
        """
        super().__init__(context)
        # 保存配置
        self.config = config
        
        # 用于存储每个会话的游戏状态
        # 键为 event.unified_msg_origin，值为 dict: {"game_active": bool, "llm_context": list, "last_options": dict}
        self.game_sessions: Dict[str, Dict[str, Any]] = {}

        # 提示词模板属性名称
        self.SYSTEM_SCENE_PROMPT_NAME = "SYSTEM_SCENE_PROMPT"
        self.OPTION_A_PROMPT_NAME = "OPTION_A_PROMPT"
        self.OPTION_B_PROMPT_NAME = "OPTION_B_PROMPT"
        self.OPTION_C_PROMPT_NAME = "OPTION_C_PROMPT"
        self.SYSTEM_RESPONSE_PROMPT_NAME = "SYSTEM_RESPONSE_PROMPT"

        # 从配置中加载提示词模板
        self.prompt_templates = {
            self.SYSTEM_SCENE_PROMPT_NAME: self.config.get("prompts", {}).get(
                "scene_prompt", 
                "你现在扮演Galgame中的一个角色，请根据当前人格设定，以第一人称视角创造一个沉浸式开场：1)描述周围环境和氛围，2)表达你(角色)此刻的心情和想法，3)向玩家(称为'你')自然地开启对话。注意保持角色特点一致，并在对话中埋下后续剧情的伏笔。"
            ),
            self.OPTION_A_PROMPT_NAME: self.config.get("prompts", {}).get(
                "option_a_prompt",
                "基于当前故事情境，为玩家创建一个温柔/体贴/善解人意风格的互动选项，标记为A。这个选项应该是玩家对角色说的话或采取的行动，而非角色的想法。必须严格按照'A - [选项内容]'格式输出，内容控制在20字以内。"
            ),
            self.OPTION_B_PROMPT_NAME: self.config.get("prompts", {}).get(
                "option_b_prompt",
                "基于当前故事情境，为玩家创建一个挑逗/暧昧/幽默风格的互动选项，标记为B。这个选项应该是玩家对角色说的话或采取的行动，而非角色的想法。必须严格按照'B - [选项内容]'格式输出，内容控制在20字以内。"
            ),
            self.OPTION_C_PROMPT_NAME: self.config.get("prompts", {}).get(
                "option_c_prompt",
                "基于当前故事情境，为玩家创建一个理性/保守/谨慎风格的互动选项，标记为C。这个选项应该是玩家对角色说的话或采取的行动，而非角色的想法。必须严格按照'C - [选项内容]'格式输出，内容控制在20字以内。"
            ),
            self.SYSTEM_RESPONSE_PROMPT_NAME: self.config.get("prompts", {}).get(
                "response_prompt",
                "玩家已选择了一个互动选项。请你以角色视角，根据玩家的选择自然地延续对话和情节。回应中应该：1)表现出角色对玩家选择的情感反应，2)推进故事情节发展，3)展示角色的个性特点，4)留下悬念以便故事继续。保持叙述生动且符合角色设定。"
            )
        }

        # 获取启用的群聊列表
        self.enabled_groups = set(self.config.get("enabled_groups", []))
        
        # 获取自定义提供商ID
        self.llm_provider_id = self.config.get("llm_provider_id", "")
        
        # 验证提供商ID是否可用
        self._verify_provider_id()
        
        logger.info("GalGame 插件初始化完成")
        
    def _verify_provider_id(self):
        """
        验证配置的提供商ID是否有效，如果无效则记录警告
        """
        if not self.llm_provider_id:
            logger.warning("未配置LLM提供商ID，将尝试使用默认提供商")
            return
            
        provider = self.context.get_provider_by_id(self.llm_provider_id)
        if not provider:
            logger.warning(f"指定的LLM提供商ID '{self.llm_provider_id}' 不存在，将尝试使用默认提供商")
        else:
            logger.info(f"已成功配置LLM提供商: {self.llm_provider_id}")

    def _get_llm_provider(self):
        """
        获取用于游戏生成的LLM提供商
        如果配置了有效的提供商ID，则使用该提供商
        否则回退到默认提供商
        """
        if self.llm_provider_id:
            provider = self.context.get_provider_by_id(self.llm_provider_id)
            if provider:
                return provider
                
            # 如果指定的提供商不存在，记录警告
            logger.warning(f"无法获取指定的LLM提供商 '{self.llm_provider_id}'，回退到默认提供商")
                
        # 回退到默认提供商
        default_provider = self.context.get_using_provider()
        if not default_provider:
            logger.error("无法获取默认LLM提供商，请检查是否启用了大语言模型")
            
        return default_provider

    def _get_system_prompt(self, persona_id: Optional[str], default_prompt: str) -> str:
        '''获取系统提示词'''
        try:
            if persona_id is None:
                # 使用默认人格
                default_persona = self.context.provider_manager.selected_default_persona
                if default_persona:
                    return default_persona.get("prompt", default_prompt)
            elif persona_id != "[%None]":
                # 使用指定人格
                personas = self.context.provider_manager.personas
                for persona in personas:
                    if persona.get("id") == persona_id:
                        return persona.get("prompt", default_prompt)
        except Exception as e:
            logger.error(f"获取人格信息时出错: {str(e)}")

        return default_prompt

    def _check_group_permitted(self, event: AstrMessageEvent) -> bool:
        """
        检查群聊是否允许运行游戏
        :param event: AstrBot 消息事件
        :return: 是否允许
        """
        # 如果是私聊，始终允许
        if event.is_private_chat():
            return True
            
        # 如果是群聊，检查是否在允许的群列表中
        group_id = event.get_group_id()
        if not group_id:
            return True  # 无法获取群ID，默认允许
            
        # 如果启用群列表为空，允许所有群
        if not self.enabled_groups:
            return True
            
        # 检查群ID是否在允许列表中
        return group_id in self.enabled_groups

    @filter.command("gal启动", priority=1)
    async def handle_start_galgame(self, event: AstrMessageEvent):
        """
        启动 Galgame 游戏。
        :param event: AstrBot 消息事件
        """
        # 检查是否在允许的群中
        if not self._check_group_permitted(event):
            yield event.plain_result("该群聊未启用gal游戏功能")
            return
            
        # 检查LLM提供商是否可用
        provider = self._get_llm_provider()
        if not provider:
            yield event.plain_result("无法获取LLM提供商，请联系管理员")
            return
            
        session_id = event.unified_msg_origin
        # 检查是否已有活跃会话
        if session_id in self.game_sessions and self.game_sessions[session_id].get("game_active", False):
            yield event.plain_result("已经有一个进行中的 gal 游戏，请先使用 'gal关闭' 结束当前游戏")
            return
        # 初始化/重置会话状态
        self.game_sessions[session_id] = {
            "game_active": True,
            "llm_context": [],
            "last_options": {}
        }
        # 告知用户游戏已启动
        yield event.plain_result("gal已启动")
        # 生成开局场景和第一组选项
        async for response in self._generate_initial_scene(event):
            yield response

    @filter.command("gal关闭", priority=1)
    async def handle_stop_galgame(self, event: AstrMessageEvent):
        """
        关闭 Galgame 游戏。
        :param event: AstrBot 消息事件
        """
        # 检查是否在允许的群中
        if not self._check_group_permitted(event):
            yield event.plain_result("该群聊未启用gal游戏功能")
            return
            
        session_id = event.unified_msg_origin
        # 检查是否有活跃会话
        if session_id not in self.game_sessions or not self.game_sessions[session_id].get("game_active", False):
            yield event.plain_result("当前没有进行中的 gal 游戏")
            return
        # 标记为非激活并清空上下文
        self.game_sessions[session_id]["game_active"] = False
        self.game_sessions[session_id]["llm_context"] = []
        # 告知用户游戏已关闭
        yield event.plain_result("gal已关闭")

    @filter.event_message_type(filter.EventMessageType.ALL, priority=0)
    async def handle_game_input(self, event: AstrMessageEvent):
        """
        处理游戏中的用户输入（A/B/C），拦截并处理选项选择。
        :param event: AstrBot 消息事件
        """
        # 检查是否在允许的群中
        if not self._check_group_permitted(event):
            return
            
        session_id = event.unified_msg_origin
        # 检查该会话是否有活跃的游戏
        if session_id in self.game_sessions and self.game_sessions[session_id].get("game_active", False):
            user_input = event.message_str.strip().upper()
            # 处理选项选择
            if user_input in ["A", "B", "C"]:
                # 阻止默认的LLM处理
                event.should_call_llm(False)
                # 处理用户选择
                async for response in self._process_user_choice(event, user_input):
                    yield response
            elif user_input not in ["GAL启动", "GAL关闭"]:  # 避免与命令冲突
                # 对于非A/B/C输入，只阻止事件传播，不做任何提示
                event.stop_event()  # 阻止其他插件处理此消息

    async def _generate_initial_scene(self, event: AstrMessageEvent):
        """
        生成游戏初始场景并发送给用户，然后生成第一组选项。
        :param event: AstrBot 消息事件
        """
        session_id = event.unified_msg_origin
        try:
            # 获取当前对话ID和对话对象
            conversation_id = await self.context.conversation_manager.get_curr_conversation_id(session_id)
            conversation = await self.context.conversation_manager.get_conversation(session_id, conversation_id)
            
            if not conversation:
                logger.error("无法获取对话")
                yield event.plain_result("无法获取对话，请重新开始游戏")
                return
                
            # 获取LLM提供商
            provider = self._get_llm_provider()
            if not provider:
                yield event.plain_result("无法获取LLM提供商，请联系管理员")
                return
                
            # 获取系统提示词（结合当前人格）
            system_prompt = self._get_system_prompt(
                conversation.persona_id if hasattr(conversation, 'persona_id') else None,
                "你是一个视觉小说游戏引擎，能生成优质的Galgame剧情和选项"
            )
            
            # 使用text_chat并显式获取响应
            system_scene_prompt_text = self.prompt_templates[self.SYSTEM_SCENE_PROMPT_NAME]
            scene_response = await provider.text_chat(
                prompt=system_scene_prompt_text,
                system_prompt=system_prompt,
                contexts=self.game_sessions[session_id]["llm_context"]
            )
            
            # 获取生成的场景文本
            scene_text = scene_response.completion_text if hasattr(scene_response, 'completion_text') else "欢迎来到游戏世界"
            
            # 发送给用户
            yield event.plain_result(scene_text)
            
            # 写入上下文 - 现在包含确切的场景内容
            self.game_sessions[session_id]["llm_context"].append({"role": "system", "content": system_scene_prompt_text})
            self.game_sessions[session_id]["llm_context"].append({"role": "assistant", "content": scene_text})
            
            # 生成第一组选项
            async for response in self._generate_options(event):
                yield response
        except Exception as e:
            logger.error(f"生成初始场景时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            yield event.plain_result(f"生成场景时出错: {str(e)}")

    async def _generate_options(self, event: AstrMessageEvent):
        """
        生成三个选项并分别发送给用户，同时发送按钮。
        :param event: AstrBot 消息事件
        """
        session_id = event.unified_msg_origin
        try:
            # 获取LLM提供商
            provider = self._get_llm_provider()
            if not provider:
                yield event.plain_result("无法获取LLM提供商，请联系管理员")
                return
                
            # 注意：这里使用固定的系统提示词，不注入人格
            fixed_system_prompt = "你是一个视觉小说游戏引擎，负责生成玩家可以选择的选项"
            
            self.game_sessions[session_id]["last_options"] = {}
            
            # 生成选项A
            option_a_prompt_text = self.prompt_templates[self.OPTION_A_PROMPT_NAME]
            option_a_response = await provider.text_chat(
                prompt=option_a_prompt_text,
                system_prompt=fixed_system_prompt,
                contexts=self.game_sessions[session_id]["llm_context"]
            )
            option_a_text = option_a_response.completion_text if hasattr(option_a_response, 'completion_text') else "A - 温柔微笑"
            self.game_sessions[session_id]["last_options"]["A"] = option_a_text
            yield event.plain_result(f"{option_a_text}")
            
            # 生成选项B
            option_b_prompt_text = self.prompt_templates[self.OPTION_B_PROMPT_NAME]
            option_b_response = await provider.text_chat(
                prompt=option_b_prompt_text,
                system_prompt=fixed_system_prompt,
                contexts=self.game_sessions[session_id]["llm_context"]
            )
            option_b_text = option_b_response.completion_text if hasattr(option_b_response, 'completion_text') else "B - 挑逗一笑"
            self.game_sessions[session_id]["last_options"]["B"] = option_b_text
            yield event.plain_result(f"{option_b_text}")
            
            # 生成选项C
            option_c_prompt_text = self.prompt_templates[self.OPTION_C_PROMPT_NAME]
            option_c_response = await provider.text_chat(
                prompt=option_c_prompt_text,
                system_prompt=fixed_system_prompt,
                contexts=self.game_sessions[session_id]["llm_context"]
            )
            option_c_text = option_c_response.completion_text if hasattr(option_c_response, 'completion_text') else "C - 保持距离"
            self.game_sessions[session_id]["last_options"]["C"] = option_c_text
            yield event.plain_result(f"{option_c_text}")
            
            # 创建按钮数据结构
            buttons = {
                "type": "button",
                "content": [
                    [
                        {"label": "A", "callback": "A"},
                        {"label": "B", "callback": "B"},
                        {"label": "C", "callback": "C"}
                    ]
                ]
            }
            
            # 将按钮以字典格式发送（astrbot_plugin_buttons 插件会自动处理）
            yield event.plain_result(f"{buttons}")
            
            # 选项写入上下文
            self.game_sessions[session_id]["llm_context"].append({
                "role": "assistant",
                "content": f"提供的选项：\n{option_a_text}\n{option_b_text}\n{option_c_text}"
            })
        except Exception as e:
            logger.error(f"生成选项时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            yield event.plain_result(f"生成选项时出错: {str(e)}")

    async def _process_user_choice(self, event: AstrMessageEvent, choice: str):
        """
        处理用户选择，生成后续剧情。
        :param event: AstrBot 消息事件
        :param choice: 用户选择的选项（A/B/C）
        """
        session_id = event.unified_msg_origin
        try:
            # 获取用户选择的选项文本
            chosen_option_full_text = self.game_sessions[session_id]["last_options"].get(choice)
            if not chosen_option_full_text:
                yield event.plain_result("无法识别您的选择，请重新选择A、B或C")
                return
                
            # 记录用户选择到对话历史
            self.game_sessions[session_id]["llm_context"].append({
                "role": "user",
                "content": f"用户选择了：{chosen_option_full_text}"
            })
            
            # 生成后续剧情
            async for response in self._generate_story_progression(event, choice, chosen_option_full_text):
                yield response
        except Exception as e:
            logger.error(f"处理用户选择时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            yield event.plain_result(f"处理选择时出错: {str(e)}")

    async def _generate_story_progression(self, event: AstrMessageEvent, choice: str, chosen_option_full_text: str):
        """
        根据用户选择生成故事进展并发送给用户，然后生成新一轮选项。
        :param event: AstrBot 消息事件
        :param choice: 用户选择的选项（A/B/C）
        :param chosen_option_full_text: 选项的完整文本
        """
        session_id = event.unified_msg_origin
        try:
            # 获取当前对话ID和对话对象
            conversation_id = await self.context.conversation_manager.get_curr_conversation_id(session_id)
            conversation = await self.context.conversation_manager.get_conversation(session_id, conversation_id)
            
            if not conversation:
                logger.error("无法获取对话")
                yield event.plain_result("无法获取对话，请重新开始游戏")
                return
                
            # 获取LLM提供商
            provider = self._get_llm_provider()
            if not provider:
                yield event.plain_result("无法获取LLM提供商，请联系管理员")
                return
                
            # 获取系统提示词（结合当前人格）
            system_prompt = self._get_system_prompt(
                conversation.persona_id if hasattr(conversation, 'persona_id') else None,
                "你是一个视觉小说游戏引擎，能根据用户选择生成优质的Galgame剧情"
            )
            
            system_response_prompt_text = self.prompt_templates[self.SYSTEM_RESPONSE_PROMPT_NAME]
            
            # 构建提示词
            prompt = f"{system_response_prompt_text}\n玩家选择: {chosen_option_full_text}"
            
            # 使用text_chat显式获取响应
            story_response = await provider.text_chat(
                prompt=prompt,
                system_prompt=system_prompt,
                contexts=self.game_sessions[session_id]["llm_context"]
            )
            story_text = story_response.completion_text if hasattr(story_response, 'completion_text') else "故事继续..."
            
            # 发送响应给用户
            yield event.plain_result(story_text)
            
            # 更新上下文 - 保存确切的故事内容
            self.game_sessions[session_id]["llm_context"].append({
                "role": "assistant", 
                "content": story_text
            })
            
            # 继续生成新选项
            async for option_response in self._generate_options(event):
                yield option_response
        except Exception as e:
            logger.error(f"生成故事进展时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            yield event.plain_result(f"生成故事进展时出错: {str(e)}")

    async def terminate(self):
        """
        插件终止时清理资源
        """
        self.game_sessions.clear()
        logger.info("GalGame 插件已终止")
