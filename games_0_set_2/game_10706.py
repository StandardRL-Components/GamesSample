import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:55:06.571267
# Source Brief: brief_00706.md
# Brief Index: 706
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a rhythmic block stacking game.
    The goal is to stack falling blocks as high as possible, synchronizing
    placements with a beat for bonus points. The tower's stability depends
    on how well the blocks overlap.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Stack falling blocks as high as you can. Place blocks on the beat to score bonus points and build a stable tower."
    )
    user_guide = (
        "Use the ← and → arrow keys to move the falling block. Press space to drop the block quickly and place it."
    )
    auto_advance = True


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30

        # Colors
        self.COLOR_BG = (15, 18, 28)
        self.COLOR_GRID = (30, 35, 50)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.BLOCK_COLORS = [
            (231, 76, 60),   # Red
            (52, 152, 219),  # Blue
            (46, 204, 113),  # Green
            (241, 196, 15),  # Yellow
            (155, 89, 182),  # Purple
        ]
        self.BEAT_COLOR_GOOD = (0, 255, 128)
        self.BEAT_COLOR_BAD = (70, 70, 90)

        # Game Parameters
        self.BLOCK_WIDTH = 100
        self.BLOCK_HEIGHT = 25
        self.BASE_FALL_SPEED = 2.0
        self.MOVE_SPEED = 10
        self.BEAT_PERIOD = 45
        self.GOOD_BEAT_WINDOW = 0.9
        self.STABILITY_THRESHOLD = 0.8
        self.MAX_STEPS = 1500
        self.WIN_SCORE = 500

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 36, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stacked_blocks = []
        self.falling_block_rect = None
        self.falling_block_color = None
        self.fall_speed = self.BASE_FALL_SPEED
        self.last_space_held = False
        self.particles = []
        self.color_index = 0
        
        # self.reset() # Removed to follow Gymnasium API, reset is called by user
        # self.validate_implementation() # Removed, not part of standard env

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.fall_speed = self.BASE_FALL_SPEED
        self.last_space_held = False
        self.particles = []
        self.color_index = 0

        # Create a wide, stable base block
        base_rect = pygame.Rect(
            (self.WIDTH - self.BLOCK_WIDTH * 2) / 2,
            self.HEIGHT - self.BLOCK_HEIGHT,
            self.BLOCK_WIDTH * 2,
            self.BLOCK_HEIGHT,
        )
        self.stacked_blocks = [(base_rect, (100, 100, 110))]

        self._spawn_falling_block()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # --- Action Handling ---
        movement = action[0]
        space_held = action[1] == 1
        space_pressed = space_held and not self.last_space_held
        self.last_space_held = space_held

        # --- Game Logic ---
        if self.falling_block_rect:
            # 1. Move falling block horizontally
            if movement == 3:  # Left
                self.falling_block_rect.x -= self.MOVE_SPEED
            elif movement == 4:  # Right
                self.falling_block_rect.x += self.MOVE_SPEED
            
            self.falling_block_rect.left = max(0, self.falling_block_rect.left)
            self.falling_block_rect.right = min(self.WIDTH, self.falling_block_rect.right)

            # 2. Fast drop on space press
            if space_pressed:
                # # Sound effect: Whoosh
                top_of_stack = self.stacked_blocks[-1][0].top
                self.falling_block_rect.bottom = top_of_stack
                placement_reward = self._place_block()
                reward += placement_reward
            else:
                # 3. Normal gravity
                self.falling_block_rect.y += self.fall_speed
                
                # 4. Check for automatic placement (collision)
                top_of_stack = self.stacked_blocks[-1][0].top
                if self.falling_block_rect.bottom >= top_of_stack:
                    self.falling_block_rect.bottom = top_of_stack
                    placement_reward = self._place_block()
                    reward += placement_reward
        
        self._update_particles()
        if self.steps > 0 and self.steps % 500 == 0:
            self.fall_speed += 0.05
        
        terminated = self.game_over
        
        if self.score >= self.WIN_SCORE and not terminated:
            reward += 100
            terminated = True
            self.game_over = True
            # # Sound effect: Victory Fanfare
            
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
            self.game_over = True

        truncated = False # Per Gymnasium API, truncated is for time limits, not game over
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _place_block(self):
        reward = 0
        base_block_rect, _ = self.stacked_blocks[-1]
        
        intersection = self.falling_block_rect.clip(base_block_rect)
        overlap_ratio = intersection.width / self.falling_block_rect.width if self.falling_block_rect.width > 0 else 0

        if overlap_ratio >= self.STABILITY_THRESHOLD:
            # # Sound effect: Block Place Click
            self.falling_block_rect.x = intersection.x
            self.falling_block_rect.width = intersection.width
            
            self.stacked_blocks.append((self.falling_block_rect, self.falling_block_color))
            
            reward += 1
            self.score += 1
            
            if self._is_on_beat():
                reward += 2
                self.score += 2
                # # Sound effect: Beat Sync Blip
                self._create_particles(self.falling_block_rect.midbottom, self.BEAT_COLOR_GOOD, 25, is_beat_sync=True)
            else:
                reward -= 0.5 # Less harsh penalty
                # # Sound effect: Missed Beat Thud
            
            if (len(self.stacked_blocks) - 1) % 10 == 0 and len(self.stacked_blocks) > 1:
                reward += 10
                self.score += 10
                # # Sound effect: Bonus Chime
            
            self._create_particles(self.falling_block_rect.midbottom, self.falling_block_color, 20)
            
            if self.falling_block_rect.top <= self.BLOCK_HEIGHT:
                self.game_over = True
            else:
                self._spawn_falling_block()
        else:
            # # Sound effect: Stack Collapse Crash
            self.game_over = True
            reward -= 10
            self._create_particles(self.falling_block_rect.center, (200, 200, 220), 100, is_collapse=True)
            self.falling_block_rect = None

        return reward

    def _spawn_falling_block(self):
        self.color_index = (self.color_index + 1) % len(self.BLOCK_COLORS)
        self.falling_block_color = self.BLOCK_COLORS[self.color_index]
        
        last_block_x = self.stacked_blocks[-1][0].centerx
        spawn_x = last_block_x - self.BLOCK_WIDTH / 2
        
        self.falling_block_rect = pygame.Rect(
            spawn_x, 0, self.stacked_blocks[-1][0].width, self.BLOCK_HEIGHT
        )

    def _is_on_beat(self):
        beat_phase = (self.steps % self.BEAT_PERIOD) / self.BEAT_PERIOD
        return beat_phase >= self.GOOD_BEAT_WINDOW

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_particles()
        
        for rect, color in self.stacked_blocks:
            self._render_block(rect, color)
        if self.falling_block_rect:
            self._render_block(self.falling_block_rect, self.falling_block_color, is_falling=True)
            ghost_rect = self.falling_block_rect.copy()
            ghost_rect.bottom = self.stacked_blocks[-1][0].top
            self._render_block(ghost_rect, (*self.falling_block_color, 60), is_ghost=True)

        self._render_beat_indicator()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}
    
    def _render_grid(self):
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_block(self, rect, color, is_falling=False, is_ghost=False):
        if is_ghost:
            pygame.gfxdraw.box(self.screen, rect, color)
            return

        pygame.draw.rect(self.screen, color, rect, border_radius=3)
        
        highlight_color = tuple(min(255, c + 40) for c in color)
        highlight_rect = pygame.Rect(rect.left, rect.top, rect.width, int(rect.height * 0.25))
        pygame.draw.rect(self.screen, highlight_color, highlight_rect, border_top_left_radius=3, border_top_right_radius=3)

        if is_falling:
            glow_surface = pygame.Surface((rect.width + 20, rect.height + 20), pygame.SRCALPHA)
            pygame.draw.rect(glow_surface, (*color, 60), glow_surface.get_rect(), border_radius=12)
            self.screen.blit(glow_surface, (rect.x - 10, rect.y - 10), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_beat_indicator(self):
        beat_phase = (self.steps % self.BEAT_PERIOD) / self.BEAT_PERIOD
        pulse = abs(math.sin(beat_phase * math.pi))
        
        center_x, base_y = self.WIDTH // 2, self.HEIGHT - 30
        color = self.BEAT_COLOR_GOOD if beat_phase >= self.GOOD_BEAT_WINDOW else self.BEAT_COLOR_BAD
        radius = int(10 + pulse * 15)
        
        pygame.gfxdraw.filled_circle(self.screen, center_x, base_y, radius, (*color, 50))
        pygame.gfxdraw.filled_circle(self.screen, center_x, base_y, int(radius * 0.7), color)
        pygame.gfxdraw.aacircle(self.screen, center_x, base_y, int(radius * 0.7), color)

    def _render_ui(self):
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))
        goal_text = self.font_small.render(f"GOAL: {self.WIN_SCORE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(goal_text, (20, 50))
        
    def _create_particles(self, pos, color, count, is_collapse=False, is_beat_sync=False):
        for _ in range(count):
            if is_collapse:
                angle, speed = random.uniform(0, 2 * math.pi), random.uniform(2, 8)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                life, radius = random.randint(40, 70), random.uniform(3, 8)
            elif is_beat_sync:
                angle, speed = random.uniform(-math.pi*0.8, -math.pi*0.2), random.uniform(2, 5)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                life, radius = random.randint(30, 50), random.uniform(2, 5)
            else:
                angle, speed = random.uniform(-math.pi*0.75, -math.pi*0.25), random.uniform(1, 3)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                life, radius = random.randint(20, 40), random.uniform(1, 4)
            self.particles.append(
                {"pos": list(pos), "vel": vel, "life": life, "max_life": life, "color": color, "radius": radius}
            )

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _render_particles(self):
        for p in self.particles:
            life_ratio = p["life"] / p["max_life"]
            radius = int(p["radius"] * life_ratio)
            if radius > 0:
                pos = (int(p["pos"][0]), int(p["pos"][1]))
                color = (*p["color"], int(255 * life_ratio))
                temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (radius, radius), radius)
                self.screen.blit(temp_surf, (pos[0] - radius, pos[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # Set the video driver to a real one for visualization
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Stacker")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space = 1
        
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Score: {info['score']}, Reward: {total_reward:.2f}")
            total_reward = 0
            obs, info = env.reset()
            pygame.time.wait(2000)

        clock.tick(env.FPS)
    env.close()