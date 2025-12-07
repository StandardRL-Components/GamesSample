import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:55:12.578549
# Source Brief: brief_00090.md
# Brief Index: 90
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Stack falling blocks as high as you can to build a stable tower. "
        "Place blocks carefully to avoid a catastrophic collapse!"
    )
    user_guide = (
        "Controls: ←→ to move the falling block. Hold space to drop the block faster."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GAME_AREA_WIDTH = 240
    MAX_STEPS = 9000  # 300 seconds * 30 FPS
    GAME_DURATION = 300.0 # seconds

    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_GRID = (45, 48, 56)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_GAMEOVER_OVERLAY = (25, 28, 36, 180)
    BLOCK_COLORS = [
        (255, 95, 95),   # Red
        (95, 255, 95),   # Green
        (95, 95, 255),   # Blue
        (255, 255, 95),  # Yellow
        (95, 255, 255),  # Cyan
        (255, 95, 255),  # Magenta
    ]

    # Block & Physics
    BLOCK_WIDTH = 60
    BLOCK_HEIGHT = 20
    BASE_FALL_SPEED = 2.0
    FAST_DROP_MULTIPLIER = 10
    PLAYER_SPEED = 6.0
    MIN_SUPPORT_RATIO = 0.25 # Minimum overlap to be considered stable

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_gameover = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # --- Game State Initialization ---
        self.steps = 0
        self.score = 0
        self.time_remaining = self.GAME_DURATION
        self.game_over = False
        self.stacked_blocks = []
        self.falling_block = None
        self.particles = []
        self.fall_speed = self.BASE_FALL_SPEED
        self.blocks_placed_count = 0
        self.last_space_held = False
        self.game_area_x_start = (self.SCREEN_WIDTH - self.GAME_AREA_WIDTH) // 2
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.time_remaining = self.GAME_DURATION
        self.game_over = False
        self.particles = []
        self.fall_speed = self.BASE_FALL_SPEED
        self.blocks_placed_count = 0
        self.last_space_held = False

        # Create the base platform
        base_rect = pygame.Rect(
            self.game_area_x_start, self.SCREEN_HEIGHT - self.BLOCK_HEIGHT, 
            self.GAME_AREA_WIDTH, self.BLOCK_HEIGHT
        )
        self.stacked_blocks = [{
            "rect": base_rect,
            "color": (100, 100, 100),
            "is_base": True
        }]
        
        self._spawn_new_block()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        # --- Handle Actions ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        if movement == 3: # Left
            self.falling_block["rect"].x -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.falling_block["rect"].x += self.PLAYER_SPEED
        
        # Clamp falling block to game area
        self.falling_block["rect"].left = max(self.game_area_x_start, self.falling_block["rect"].left)
        self.falling_block["rect"].right = min(self.game_area_x_start + self.GAME_AREA_WIDTH, self.falling_block["rect"].right)
        
        # --- Game Logic ---
        self.time_remaining -= 1.0 / 30.0 # Assuming 30 FPS logic update
        self.falling_block["time_since_spawn"] += 1
        
        current_fall_speed = self.fall_speed
        if space_held:
            current_fall_speed *= self.FAST_DROP_MULTIPLIER

        self.falling_block["rect"].y += current_fall_speed
        
        # --- Collision and Landing ---
        top_block = self.stacked_blocks[-1]
        if self.falling_block["rect"].colliderect(top_block["rect"]):
            landing_y = top_block["rect"].top - self.BLOCK_HEIGHT
            self.falling_block["rect"].y = landing_y
            
            placement_reward, collapse_penalty = self._handle_block_landing()
            reward += placement_reward + collapse_penalty

        # --- Continuous Reward for Stability ---
        if self._is_falling_block_stable_hover():
            reward += 1

        # --- Update Particles ---
        self._update_particles()
        
        # --- Check Termination Conditions ---
        terminated = False
        if self.score >= 1000:
            reward += 100
            terminated = True
        elif self.time_remaining <= 0:
            reward -= 100
            terminated = True
        elif len(self.stacked_blocks) == 1 and self.blocks_placed_count > 0: # Stack collapse
            reward -= 100 # Penalty for full collapse
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated
        
        # Truncated is always False as termination is part of the MDP
        truncated = False
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_block_landing(self):
        top_block = self.stacked_blocks[-1]
        
        overlap_rect = self.falling_block["rect"].clip(top_block["rect"])
        overlap_width = overlap_rect.width
        
        # Check for stability
        if overlap_width < self.BLOCK_WIDTH * self.MIN_SUPPORT_RATIO:
            # Unstable drop, block falls off
            self._create_particles(self.falling_block["rect"].centerx, self.falling_block["rect"].bottom, self.falling_block["color"], 40, is_collapse=True)
            self._spawn_new_block()
            return 0, -50 # Placement reward is 0, collapse penalty is -50 for the lost block
        else:
            # Stable placement
            self.stacked_blocks.append(self.falling_block)
            self.blocks_placed_count += 1
            
            # Calculate placement reward
            time_bonus = max(10, 50 - self.falling_block["time_since_spawn"] * 0.2)
            placement_reward = time_bonus
            self.score += int(time_bonus)

            self._create_particles(self.falling_block["rect"].centerx, self.falling_block["rect"].bottom, self.falling_block["color"], 20)
            
            # Check for chain reaction collapse
            collapse_penalty = self._check_for_collapse()
            
            # Increase difficulty
            if self.blocks_placed_count > 0 and self.blocks_placed_count % 10 == 0:
                self.fall_speed += 0.2
            
            self._spawn_new_block()
            return placement_reward, collapse_penalty

    def _check_for_collapse(self):
        collapse_penalty = 0
        for i in range(len(self.stacked_blocks) - 1, 0, -1):
            block_to_check = self.stacked_blocks[i]
            if block_to_check.get("is_base"): continue

            # Find supporting surface from the blocks below
            support_min_x = float('inf')
            support_max_x = float('-inf')
            is_supported = False
            for j in range(i - 1, -1, -1):
                lower_block = self.stacked_blocks[j]
                if lower_block["rect"].top == block_to_check["rect"].bottom:
                    if block_to_check["rect"].colliderect(lower_block["rect"]):
                        is_supported = True
                        support_min_x = min(support_min_x, lower_block["rect"].left)
                        support_max_x = max(support_max_x, lower_block["rect"].right)
            
            if not is_supported or not (support_min_x <= block_to_check["rect"].centerx <= support_max_x):
                # Block is unstable, remove it and everything above it
                num_collapsed = len(self.stacked_blocks) - i
                collapsed_blocks = self.stacked_blocks[i:]
                self.stacked_blocks = self.stacked_blocks[:i]
                
                for block in collapsed_blocks:
                    self._create_particles(block["rect"].centerx, block["rect"].centery, block["color"], 30, is_collapse=True)
                
                collapse_penalty = -50 * num_collapsed
                self.score -= 50 * num_collapsed
                break # Stop checking after the first collapse is found
        return collapse_penalty

    def _spawn_new_block(self):
        start_x = self.np_random.integers(
            self.game_area_x_start, 
            self.game_area_x_start + self.GAME_AREA_WIDTH - self.BLOCK_WIDTH + 1
        )
        rect = pygame.Rect(start_x, -self.BLOCK_HEIGHT, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
        color_index = self.np_random.integers(0, len(self.BLOCK_COLORS))
        color = self.BLOCK_COLORS[color_index]
        self.falling_block = {
            "rect": rect,
            "color": color,
            "time_since_spawn": 0
        }

    def _is_falling_block_stable_hover(self):
        if not self.falling_block or not self.stacked_blocks:
            return False
        
        top_block_rect = self.stacked_blocks[-1]["rect"]
        # Check if horizontally aligned
        return (self.falling_block["rect"].left < top_block_rect.right and 
                self.falling_block["rect"].right > top_block_rect.left)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(self.game_area_x_start, self.game_area_x_start + self.GAME_AREA_WIDTH + 1, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.game_area_x_start, y), (self.game_area_x_start + self.GAME_AREA_WIDTH, y))
        
        # Draw game area boundaries
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.game_area_x_start, 0, self.GAME_AREA_WIDTH, self.SCREEN_HEIGHT), 2)

        # Draw particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), int(p['size']), p['color'])

        # Draw stacked blocks
        for block in self.stacked_blocks:
            self._draw_block(block)

        # Draw ghost piece
        if self.falling_block and not self.game_over:
            self._draw_ghost_block()
            
        # Draw falling block
        if self.falling_block and not self.game_over:
            self._draw_block(self.falling_block)

    def _draw_block(self, block_data):
        rect = block_data["rect"]
        color = block_data["color"]
        border_color = tuple(max(0, c - 50) for c in color)
        
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, border_color, rect, 2)

    def _draw_ghost_block(self):
        ghost_rect = self.falling_block["rect"].copy()
        
        # Raycast down to find landing position
        top_block = self.stacked_blocks[-1]
        ghost_rect.y = top_block["rect"].top - self.BLOCK_HEIGHT

        # Create a transparent surface for the ghost
        ghost_surface = pygame.Surface((ghost_rect.width, ghost_rect.height), pygame.SRCALPHA)
        color = self.falling_block["color"]
        ghost_color = (*color, 70) # Add alpha
        border_color = (*tuple(max(0, c - 50) for c in color), 120)

        pygame.draw.rect(ghost_surface, ghost_color, ghost_surface.get_rect())
        pygame.draw.rect(ghost_surface, border_color, ghost_surface.get_rect(), 2)
        self.screen.blit(ghost_surface, ghost_rect.topleft)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_str = f"TIME: {max(0, int(self.time_remaining))}"
        timer_text = self.font_ui.render(time_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))
        
        # Blocks stacked
        level_text = self.font_ui.render(f"BLOCKS: {self.blocks_placed_count}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (10, self.SCREEN_HEIGHT - level_text.get_height() - 10))

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_GAMEOVER_OVERLAY)
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= 1000:
                msg = "YOU WIN!"
            elif self.time_remaining <= 0:
                msg = "TIME UP!"
            else:
                msg = "TOWER COLLAPSED!"
                
            gameover_msg = self.font_gameover.render(msg, True, self.COLOR_UI_TEXT)
            msg_rect = gameover_msg.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            self.screen.blit(gameover_msg, msg_rect)

            final_score_msg = self.font_ui.render(f"Final Score: {self.score}", True, self.COLOR_UI_TEXT)
            final_rect = final_score_msg.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 30))
            self.screen.blit(final_score_msg, final_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "blocks_placed": self.blocks_placed_count,
            "fall_speed": self.fall_speed
        }
        
    def _create_particles(self, x, y, color, count, is_collapse=False):
        for _ in range(count):
            if is_collapse:
                vx = self.np_random.uniform(-4, 4)
                vy = self.np_random.uniform(-5, 2)
            else:
                vx = self.np_random.uniform(-2, 2)
                vy = self.np_random.uniform(-3, 0)
            
            self.particles.append({
                'x': x, 'y': y, 'vx': vx, 'vy': vy,
                'color': color,
                'size': self.np_random.uniform(2, 5),
                'life': self.np_random.uniform(20, 40)
            })

    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.2 # Gravity on particles
            p['life'] -= 1
            p['size'] = max(0, p['size'] - 0.1)
        self.particles = [p for p in self.particles if p['life'] > 0]

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will create a window and render the game
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Create a window to display the game
    pygame.display.set_caption("Tower Stacker")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    while running:
        # --- Human Controls ---
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] or keys[pygame.K_DOWN] else 0
        shift_held = 0 # Not used

        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and terminated:
                    obs, info = env.reset()
                    terminated = False
                elif event.key == pygame.K_ESCAPE:
                    running = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()