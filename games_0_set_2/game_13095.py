import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:55:07.030993
# Source Brief: brief_03095.md
# Brief Index: 3095
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a fast-paced, color-matching block stacking game.

    The player controls a falling block, moving it horizontally to align with
    stacks of the same color below. A successful stack increases the score and
    the game's speed. Mismatches or missed blocks lead to penalties. The goal
    is to reach a score of 100 before accumulating 3 failures.

    The visual design is clean and minimalist, with a focus on high-contrast
    elements, smooth animations, and satisfying particle effects for clear
    feedback.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Stack falling blocks into their matching color columns. Score points for correct matches, "
        "but be careful, as the game speeds up and mistakes are costly."
    )
    user_guide = (
        "Use the ← and → arrow keys to move the falling block. Press the space bar to drop it into the column below."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    BLOCK_SIZE = 40
    PLAYER_SPEED = 15
    NUM_COLUMNS = 3
    COLUMN_WIDTH = SCREEN_WIDTH / NUM_COLUMNS

    # --- Colors ---
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (30, 35, 50)
    COLOR_UI = (220, 220, 220)
    COLOR_OVERLAY = (20, 25, 40, 180) # Semi-transparent for game over
    COLORS = {
        "RED": (255, 80, 80),
        "GREEN": (80, 255, 80),
        "BLUE": (80, 120, 255)
    }
    COLOR_NAMES = list(COLORS.keys())
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 64, bold=True)
        
        # --- Game State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.failed_block_count = 0
        self.fall_speed = 0.0
        self.falling_block = None
        self.stacked_blocks = []
        self.particles = []
        self.last_space_held = False
        self.win_condition = False
        
        # self.reset() is called by the wrapper, no need to call it here.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        self.failed_block_count = 0
        self.fall_speed = 2.0
        self.last_space_held = False
        
        self.stacked_blocks = []
        self.particles = []
        
        # Create base platforms
        for i, color_name in enumerate(self.COLOR_NAMES):
            self.stacked_blocks.append({
                "rect": pygame.Rect(
                    i * self.COLUMN_WIDTH + (self.COLUMN_WIDTH - self.BLOCK_SIZE) / 2,
                    self.SCREEN_HEIGHT - self.BLOCK_SIZE,
                    self.BLOCK_SIZE,
                    self.BLOCK_SIZE
                ),
                "color_name": color_name,
                "column": i
            })

        self._spawn_block()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.1  # Survival reward

        movement = action[0]
        space_held = action[1] == 1
        
        # --- Handle Player Input ---
        self._handle_movement(movement)
        
        drop_reward = self._handle_drop(space_held)
        reward += drop_reward
        
        # --- Update Game Logic ---
        if self.falling_block:
            self.falling_block["rect"].y += self.fall_speed
            
            # Check for miss (hitting the floor)
            if self.falling_block["rect"].bottom > self.SCREEN_HEIGHT:
                # sfx: miss_sound
                self._create_particles(self.falling_block["rect"].center, (100, 100, 100), 20, "fail")
                self.failed_block_count += 1
                reward -= 5.0
                self._spawn_block()

        self.steps += 1
        
        # --- Check for Termination ---
        terminated = self._check_termination()
        truncated = False
        if terminated:
            self.game_over = True
            if self.win_condition:
                reward += 100.0 # Win bonus
            else:
                reward -= 100.0 # Loss penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_movement(self, movement_action):
        if not self.falling_block:
            return

        if movement_action == 3:  # Left
            self.falling_block["rect"].x -= self.PLAYER_SPEED
        elif movement_action == 4:  # Right
            self.falling_block["rect"].x += self.PLAYER_SPEED
        
        # Clamp position to screen bounds
        self.falling_block["rect"].x = max(0, self.falling_block["rect"].x)
        self.falling_block["rect"].right = min(self.SCREEN_WIDTH, self.falling_block["rect"].right)

    def _handle_drop(self, space_held):
        reward = 0
        space_pressed = space_held and not self.last_space_held
        self.last_space_held = space_held

        if space_pressed and self.falling_block:
            # sfx: drop_sound
            column_index = int(self.falling_block["rect"].centerx // self.COLUMN_WIDTH)
            
            # Find the top block in the target column
            column_blocks = [b for b in self.stacked_blocks if b["column"] == column_index]
            top_block_y = min(b["rect"].top for b in column_blocks) if column_blocks else self.SCREEN_HEIGHT
            target_block = next((b for b in column_blocks if b["rect"].top == top_block_y), None)

            # Check for color match
            if target_block and self.falling_block["color_name"] == target_block["color_name"]:
                # --- SUCCESSFUL STACK ---
                # sfx: success_chime
                new_block_rect = self.falling_block["rect"].copy()
                new_block_rect.bottom = top_block_y
                self.stacked_blocks.append({
                    "rect": new_block_rect,
                    "color_name": self.falling_block["color_name"],
                    "column": column_index,
                })
                self.score += 1
                self.fall_speed *= 1.1
                reward += 1.0
                self._create_particles(new_block_rect.center, self.COLORS[self.falling_block["color_name"]], 30, "success")
                
            else:
                # --- FAILED STACK ---
                # sfx: failure_buzz
                self.failed_block_count += 1
                reward -= 5.0
                
                # Find impact position for particle effect
                impact_pos = (self.falling_block["rect"].centerx, top_block_y)
                self._create_particles(impact_pos, (150, 150, 150), 50, "fail")

                # Remove all blocks in this column (except the base)
                self.stacked_blocks = [
                    b for b in self.stacked_blocks if b["column"] != column_index or b["rect"].top == self.SCREEN_HEIGHT - self.BLOCK_SIZE
                ]
            
            self._spawn_block()
        return reward

    def _spawn_block(self):
        color_name = self.np_random.choice(self.COLOR_NAMES)
        self.falling_block = {
            "rect": pygame.Rect(
                (self.SCREEN_WIDTH - self.BLOCK_SIZE) / 2, 
                -self.BLOCK_SIZE, 
                self.BLOCK_SIZE, 
                self.BLOCK_SIZE
            ),
            "color_name": color_name,
        }

    def _check_termination(self):
        if self.score >= 100:
            self.win_condition = True
            return True
        if self.failed_block_count >= 3:
            return True
        # A truncated condition is better for time/step limits
        # if self.steps >= 1000:
        #     return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "failed_blocks": self.failed_block_count,
            "fall_speed": self.fall_speed,
        }

    def _render_game(self):
        self._draw_background_grid()
        self._update_and_render_particles()

        for block in self.stacked_blocks:
            self._draw_block(block["rect"], self.COLORS[block["color_name"]])

        if self.falling_block:
            color = self.COLORS[self.falling_block["color_name"]]
            self._draw_glow(self.falling_block["rect"].center, color, 30)
            self._draw_block(self.falling_block["rect"], color)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

        # Fails
        fail_text = self.font_main.render(f"FAILS: ", True, self.COLOR_UI)
        self.screen.blit(fail_text, (self.SCREEN_WIDTH - 160, 10))
        for i in range(3):
            color = self.COLORS["RED"] if i < self.failed_block_count else (60, 60, 60)
            pygame.draw.circle(self.screen, color, (self.SCREEN_WIDTH - 60 + i * 25, 22), 8)

        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_OVERLAY)
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win_condition else "GAME OVER"
            end_text = self.font_large.render(message, True, self.COLOR_UI)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _draw_block(self, rect, color):
        shadow_color = (max(0, color[0]-50), max(0, color[1]-50), max(0, color[2]-50))
        highlight_color = (min(255, color[0]+50), min(255, color[1]+50), min(255, color[2]+50))
        
        # Main body
        pygame.draw.rect(self.screen, color, rect, border_radius=4)
        # Outline
        pygame.draw.rect(self.screen, shadow_color, rect, width=2, border_radius=4)
        # 3D-effect highlight
        pygame.draw.line(self.screen, highlight_color, rect.topleft, rect.topright, 2)
        pygame.draw.line(self.screen, highlight_color, rect.topleft, rect.bottomleft, 2)


    def _draw_background_grid(self):
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
        # Column dividers
        for i in range(1, self.NUM_COLUMNS):
            pygame.draw.line(self.screen, (50, 60, 80), (i * self.COLUMN_WIDTH, 0), (i * self.COLUMN_WIDTH, self.SCREEN_HEIGHT), 2)

    def _draw_glow(self, pos, color, radius):
        for i in range(radius, 0, -2):
            alpha = 100 * (1 - (i / radius))**2
            glow_color = (*color, int(alpha))
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), i, glow_color)

    def _create_particles(self, pos, color, count, p_type):
        for _ in range(count):
            if p_type == "success":
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                lifespan = self.np_random.integers(15, 30)
            else: # "fail"
                vel = [self.np_random.uniform(-2, 2), self.np_random.uniform(-4, -1)]
                lifespan = self.np_random.integers(20, 40)
            
            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "lifespan": lifespan,
                "max_lifespan": lifespan,
                "color": color,
            })
    
    def _update_and_render_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1  # Gravity
            p["lifespan"] -= 1

            if p["lifespan"] <= 0:
                self.particles.pop(i)
            else:
                alpha = 255 * (p["lifespan"] / p["max_lifespan"])
                radius = 4 * (p["lifespan"] / p["max_lifespan"])
                # Create a temporary surface for the particle to handle alpha correctly
                particle_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(
                    particle_surf,
                    (*p["color"], alpha),
                    (radius, radius),
                    radius
                )
                self.screen.blit(particle_surf, (int(p["pos"][0] - radius), int(p["pos"][1] - radius)))

    def close(self):
        pygame.quit()
        
if __name__ == "__main__":
    # --- Manual Play Loop ---
    # This block will not run in a headless environment. It's for local testing with a display.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Use a display screen for manual play
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Stacker")
    
    total_reward = 0
    
    while not terminated:
        # --- Action Mapping for Human Player ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # --- Rendering to Display ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Limit to 30 FPS

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    
    # Keep the window open for a bit to see the final screen
    end_time = pygame.time.get_ticks()
    while pygame.time.get_ticks() - end_time < 3000:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
        
    env.close()