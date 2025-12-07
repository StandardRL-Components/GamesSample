
# Generated: 2025-08-28T05:56:24.963420
# Source Brief: brief_02777.md
# Brief Index: 2777

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to push all blocks simultaneously. "
        "Objective: Move the colored blocks into their matching target zones before the time runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, minimalist puzzle game. Strategically push all blocks on the grid to "
        "solve the puzzle against the clock."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.render_mode = render_mode

        # Game constants
        self.GRID_WIDTH = 12
        self.GRID_HEIGHT = 8
        self.CELL_SIZE = 40
        self.WORLD_WIDTH = self.GRID_WIDTH * self.CELL_SIZE
        self.WORLD_HEIGHT = self.GRID_HEIGHT * self.CELL_SIZE
        self.X_OFFSET = (640 - self.WORLD_WIDTH) // 2
        self.Y_OFFSET = (400 - self.WORLD_HEIGHT) // 2
        self.FPS = 30
        self.TIME_LIMIT_SECONDS = 60
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID = (40, 45, 50)
        self.COLORS = {
            "red": (255, 80, 80),
            "green": (80, 255, 80),
            "blue": (80, 150, 255),
            "yellow": (255, 255, 80),
        }
        self.COLOR_NAMES = list(self.COLORS.keys())
        self.COLOR_UI_TEXT = (220, 220, 220)

        # Fonts
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_msg = pygame.font.SysFont("Verdana", 50, bold=True)

        # Game state variables (initialized in reset)
        self.blocks = []
        self.targets = []
        self.time_remaining_frames = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_state = False
        self.np_random = None

        # Animation state
        self.is_animating = False
        self.animation_frames = 0
        self.animation_duration = 6  # frames for smoother animation
        self.block_anim_data = []
        
        # Initialize state variables by calling reset
        # This is necessary before validation
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.time_remaining_frames = self.TIME_LIMIT_SECONDS * self.FPS
        
        self.is_animating = False
        self.animation_frames = 0
        self.block_anim_data.clear()

        self._generate_puzzle()
        
        return self._get_observation(), self._get_info()

    def _generate_puzzle(self):
        self.blocks.clear()
        self.targets.clear()
        
        all_positions = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        self.np_random.shuffle(all_positions)

        for i in range(len(self.COLOR_NAMES)):
            color_name = self.COLOR_NAMES[i]
            
            target_pos = all_positions.pop()
            self.targets.append({"pos": target_pos, "color": color_name, "filled": False})
            
            block_pos = all_positions.pop()
            visual_pos = (block_pos[0] * self.CELL_SIZE, block_pos[1] * self.CELL_SIZE)
            self.blocks.append({"pos": block_pos, "color": color_name, "visual_pos": visual_pos})

    def step(self, action):
        was_game_over = self.game_over
        reward = -0.1  # Per-step penalty

        if self.is_animating:
            self._update_animation()
        else:
            movement = action[0]
            if movement != 0:
                # _handle_push returns the +1 reward for each block placed
                push_reward = self._handle_push(movement)
                reward += push_reward
        
        self.time_remaining_frames = max(0, self.time_remaining_frames - 1)
        self.steps += 1

        terminated = self._check_termination()

        # Add terminal reward only on the frame termination happens
        if terminated and not was_game_over:
            if self.win_state:
                reward += 50
            else:  # Time ran out
                reward += -50

        # Also terminate at max steps
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
            if not was_game_over:
                reward += -50 # Treat as a loss
            self.game_over = True
            self.win_state = False

        self.score = self._calculate_current_score()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_push(self, movement_action):
        if self.is_animating:
            return 0

        direction, sort_key, sort_reverse = {
            1: ((0, -1), lambda b: b["pos"][1], False),  # Up
            2: ((0, 1), lambda b: b["pos"][1], True),   # Down
            3: ((-1, 0), lambda b: b["pos"][0], False), # Left
            4: ((1, 0), lambda b: b["pos"][0], True),   # Right
        }[movement_action]

        sorted_blocks = sorted(self.blocks, key=sort_key, reverse=sort_reverse)
        
        moved = False
        self.block_anim_data.clear()
        block_positions = {b["pos"] for b in self.blocks}

        for block in sorted_blocks:
            start_grid_pos = block["pos"]
            current_pos = start_grid_pos

            while True:
                next_pos = (current_pos[0] + direction[0], current_pos[1] + direction[1])
                if not (0 <= next_pos[0] < self.GRID_WIDTH and 0 <= next_pos[1] < self.GRID_HEIGHT):
                    break
                if next_pos in block_positions:
                    break
                current_pos = next_pos
            
            if current_pos != start_grid_pos:
                moved = True
                block_positions.remove(start_grid_pos)
                block_positions.add(current_pos)
                
                start_pixel_pos = (start_grid_pos[0] * self.CELL_SIZE, start_grid_pos[1] * self.CELL_SIZE)
                end_pixel_pos = (current_pos[0] * self.CELL_SIZE, current_pos[1] * self.CELL_SIZE)
                self.block_anim_data.append({"block": block, "start": start_pixel_pos, "end": end_pixel_pos})
                block["pos"] = current_pos

        if moved:
            self.is_animating = True
            self.animation_frames = 0
            # Sound placeholder: play_sound("push")

        placement_reward = 0
        blocks_to_remove = []
        for block in self.blocks:
            for target in self.targets:
                if not target["filled"] and block["pos"] == target["pos"] and block["color"] == target["color"]:
                    target["filled"] = True
                    blocks_to_remove.append(block)
                    placement_reward += 1
                    # Sound placeholder: play_sound("score")
                    break
        
        self.blocks = [b for b in self.blocks if b not in blocks_to_remove]
        
        return placement_reward

    def _update_animation(self):
        self.animation_frames += 1
        t = self.animation_frames / self.animation_duration
        progress = t * t * (3.0 - 2.0 * t)  # Smoothstep easing

        for data in self.block_anim_data:
            start_pos, end_pos = data["start"], data["end"]
            current_x = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
            current_y = start_pos[1] + (end_pos[1] - start_pos[1]) * progress
            data["block"]["visual_pos"] = (current_x, current_y)

        if self.animation_frames >= self.animation_duration:
            self.is_animating = False
            for data in self.block_anim_data:
                data["block"]["visual_pos"] = data["end"]

    def _calculate_current_score(self):
        score = sum(1 for t in self.targets if t["filled"])
        if self.game_over:
            if self.win_state:
                score += 50
            else:
                score -= 50
        score -= self.steps * 0.1
        return score

    def _check_termination(self):
        if not self.game_over:
            if all(t["filled"] for t in self.targets):
                self.game_over = True
                self.win_state = True
            elif self.time_remaining_frames <= 0:
                self.game_over = True
                self.win_state = False
        return self.game_over
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for x in range(self.GRID_WIDTH + 1):
            px = self.X_OFFSET + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.Y_OFFSET), (px, self.Y_OFFSET + self.WORLD_HEIGHT), 1)
        for y in range(self.GRID_HEIGHT + 1):
            py = self.Y_OFFSET + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.X_OFFSET, py), (self.X_OFFSET + self.WORLD_WIDTH, py), 1)

        for target in self.targets:
            color = self.COLORS[target["color"]]
            rect = pygame.Rect(
                self.X_OFFSET + target["pos"][0] * self.CELL_SIZE,
                self.Y_OFFSET + target["pos"][1] * self.CELL_SIZE,
                self.CELL_SIZE, self.CELL_SIZE
            )
            if target["filled"]:
                glow_color = (*color, 70)
                pygame.gfxdraw.box(self.screen, rect.inflate(-4, -4), glow_color)
                pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, 8, color)
                pygame.gfxdraw.aacircle(self.screen, rect.centerx, rect.centery, 8, color)
            else:
                pygame.draw.rect(self.screen, color, rect, 2, border_radius=5)

        for block in self.blocks:
            color = self.COLORS[block["color"]]
            darker_color = tuple(max(0, c - 60) for c in color)
            x, y = block["visual_pos"]
            rect = pygame.Rect(
                int(self.X_OFFSET + x + 2), int(self.Y_OFFSET + y + 2),
                self.CELL_SIZE - 4, self.CELL_SIZE - 4
            )
            pygame.draw.rect(self.screen, darker_color, rect, 0, border_radius=8)
            pygame.draw.rect(self.screen, color, rect.inflate(-6, -6), 0, border_radius=6)

    def _render_ui(self):
        seconds_left = math.ceil(self.time_remaining_frames / self.FPS)
        timer_text = f"TIME: {seconds_left:02d}"
        text_surf = self.font_ui.render(timer_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (640 - text_surf.get_width() - 15, 10))
        
        score_text = f"SCORE: {self.score:.0f}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (15, 10))

        if self.game_over:
            overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 190))
            self.screen.blit(overlay, (0, 0))
            
            message, color = ("COMPLETE", self.COLORS["green"]) if self.win_state else ("TIME UP", self.COLORS["red"])
            msg_surf = self.font_msg.render(message, True, color)
            self.screen.blit(msg_surf, msg_surf.get_rect(center=(320, 200)))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining_seconds": math.ceil(self.time_remaining_frames / self.FPS),
            "blocks_remaining": len(self.blocks),
        }
    
    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")