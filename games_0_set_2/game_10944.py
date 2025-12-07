import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:11:51.298253
# Source Brief: brief_00944.md
# Brief Index: 944
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a block stacking game.

    **Gameplay:**
    The player controls a horizontally moving stack of blocks at the bottom of the screen.
    Colored blocks fall from the top. The goal is to catch falling blocks that match
    the color of the topmost block on the player's stack.

    **Termination:**
    The game ends if:
    1. A falling block collides with the player stack, but the colors do not match.
    2. A falling block reaches the bottom of the screen.
    3. The maximum episode length of 5000 steps is reached.

    **Rewards:**
    - +0.1 per step for survival.
    - +10 for creating a continuous color streak of 5 blocks.
    - +1 for each block added to a streak longer than 5.
    - -100 as a terminal penalty for failure (mismatch or block hitting the floor).

    **Action Space:** `MultiDiscrete([5, 2, 2])`
    - `action[0]`: Movement (0: none, 1: up (no-op), 2: down (no-op), 3: left, 4: right)
    - `action[1]`: Space button (no-op)
    - `action[2]`: Shift button (no-op)

    **Observation Space:** `Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)`
    - An RGB image of the game screen.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a moving stack to catch falling blocks. Match the color of your top block "
        "to score points and build your stack higher."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to move the stack left and right."
    )
    auto_advance = True


    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    BLOCK_WIDTH = 40
    BLOCK_HEIGHT = 20
    PLAYER_ACCELERATION = 0.8
    PLAYER_FRICTION = 0.90
    MAX_STEPS = 5000

    # --- Colors ---
    COLOR_BG = (15, 20, 35)
    COLOR_GRID = (30, 35, 50)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_WARNING = (200, 0, 0)
    PALETTE = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 80, 255),   # Blue
        (255, 255, 80),  # Yellow
        (255, 80, 255),  # Magenta
        (80, 255, 255),  # Cyan
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)

        # --- Game State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_x = 0.0
        self.player_vel_x = 0.0
        self.player_stack = []
        self.current_streak = 0
        self.fall_speed = 0.0
        self.falling_blocks = []
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Reset Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_x = self.SCREEN_WIDTH / 2
        self.player_vel_x = 0.0
        
        # Start with a base block
        initial_color_idx = self.np_random.integers(len(self.PALETTE))
        self.player_stack = [self.PALETTE[initial_color_idx]]
        self.current_streak = 1
        
        self.fall_speed = 2.0
        self.falling_blocks = []
        self.particles = []
        
        self._spawn_falling_block()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # Although the environment should be reset, this handles lingering calls
            obs = self._get_observation()
            return obs, 0, True, False, self._get_info()

        movement = action[0]
        self.steps += 1
        
        # Base survival reward
        reward = 0.1

        # --- Update Game Logic ---
        self._update_player(movement)
        event_reward, terminated_by_logic = self._update_game_logic()
        reward += event_reward
        
        self.game_over = terminated_by_logic
        
        # --- Check Termination Conditions ---
        terminated = self.game_over
        truncated = False
        if self.steps >= self.MAX_STEPS:
            truncated = True
        
        if self.game_over:
            reward = -100.0 # Failure penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_player(self, movement):
        # Calculate acceleration based on action
        accel_x = 0.0
        if movement == 3:  # Left
            accel_x = -self.PLAYER_ACCELERATION
        elif movement == 4:  # Right
            accel_x = self.PLAYER_ACCELERATION

        # Taller stacks are harder to move (instability)
        instability_factor = 1 + (len(self.player_stack) - 1) * 0.2
        self.player_vel_x += accel_x / instability_factor
        
        # Apply friction
        self.player_vel_x *= self.PLAYER_FRICTION
        
        # Update position
        self.player_x += self.player_vel_x
        
        # Clamp to screen bounds
        half_width = self.BLOCK_WIDTH / 2
        self.player_x = np.clip(self.player_x, half_width, self.SCREEN_WIDTH - half_width)

    def _update_game_logic(self):
        event_reward = 0.0
        terminated = False

        self._update_particles()
        
        # Difficulty scaling
        if self.steps > 0 and self.steps % 500 == 0:
            self.fall_speed += 0.05

        # Update and check falling blocks
        for block in self.falling_blocks[:]:
            block['pos'].y += self.fall_speed

            # Check for floor collision
            if block['pos'].y > self.SCREEN_HEIGHT:
                self.falling_blocks.remove(block)
                terminated = True
                continue

            # Check for stack collision
            stack_top_y = self.SCREEN_HEIGHT - len(self.player_stack) * self.BLOCK_HEIGHT
            if block['pos'].y + self.BLOCK_HEIGHT / 2 > stack_top_y:
                if abs(block['pos'].x - self.player_x) < self.BLOCK_WIDTH / 2:
                    # Collision occurred
                    self.falling_blocks.remove(block)
                    
                    if block['color'] == self.player_stack[-1]:
                        # Colors match: success
                        event_reward += self._process_successful_stack(block)
                        self._spawn_particles(self.player_x, stack_top_y, block['color'])
                        self._spawn_falling_block()
                    else:
                        # Colors mismatch: failure
                        terminated = True

        return event_reward, terminated

    def _process_successful_stack(self, block):
        reward = 0.0
        
        # Check if the new block continues the color streak
        if block['color'] == self.player_stack[-1]:
            self.current_streak += 1
        else:
            # This case shouldn't happen with current rules, but is good practice
            self.current_streak = 1
            
        if self.current_streak == 5:
            reward = 10.0
        elif self.current_streak > 5:
            reward = 1.0
            
        self.player_stack.append(block['color'])
        self.score += 1
        return reward

    def _spawn_falling_block(self):
        x_pos = self.np_random.uniform(self.BLOCK_WIDTH, self.SCREEN_WIDTH - self.BLOCK_WIDTH)
        color_idx = self.np_random.integers(len(self.PALETTE))
        color = self.PALETTE[color_idx]
        
        # Increase chance of spawning a matching block
        if self.np_random.random() < 0.6: # 60% chance to match
            color = self.player_stack[-1]

        self.falling_blocks.append({
            'pos': pygame.Vector2(x_pos, -self.BLOCK_HEIGHT),
            'color': color
        })
        
    def _spawn_particles(self, x, y, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pygame.Vector2(x, y),
                'vel': vel,
                'life': self.np_random.integers(20, 40),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'].y += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        # --- Main Rendering Call ---
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        
        # Convert to numpy array and transpose for Gymnasium
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stack_height": len(self.player_stack),
            "fall_speed": self.fall_speed,
        }
        
    def _render_background(self):
        for i in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i))

    def _render_game_elements(self):
        # Warning flash
        is_warning = any(b['pos'].y > self.SCREEN_HEIGHT * 0.8 for b in self.falling_blocks)
        if is_warning:
            alpha = 50 + 40 * math.sin(self.steps * 0.2)
            warning_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT * 0.2), pygame.SRCALPHA)
            warning_surface.fill((*self.COLOR_WARNING, alpha))
            self.screen.blit(warning_surface, (0, self.SCREEN_HEIGHT * 0.8))

        # Particles
        for p in self.particles:
            size = max(1, p['life'] // 8)
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y)), size)

        # Falling blocks
        for block in self.falling_blocks:
            self._render_block(self.screen, block['pos'].x, block['pos'].y, block['color'])

        # Player stack with wobble
        wobble_amp = min(5, (len(self.player_stack) - 1) * 0.2)
        wobble = math.sin(self.steps * 0.15) * wobble_amp
        
        for i, color in enumerate(self.player_stack):
            y_pos = self.SCREEN_HEIGHT - (i + 1) * self.BLOCK_HEIGHT
            x_offset = wobble * (i / max(1, len(self.player_stack))) * math.sin(self.steps * 0.1 + i * 0.5)
            self._render_block(self.screen, self.player_x + x_offset, y_pos, color)
            
    def _render_block(self, surface, x, y, color):
        rect = pygame.Rect(0, 0, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
        rect.center = (int(x), int(y))
        
        # Darker base for 3D effect
        darker_color = tuple(max(0, c - 40) for c in color)
        pygame.draw.rect(surface, darker_color, rect, border_radius=4)
        
        # Main fill
        pygame.draw.rect(surface, color, rect.inflate(-4, -4), border_radius=3)
        
        # Highlight
        lighter_color = tuple(min(255, c + 60) for c in color)
        highlight_rect = pygame.Rect(rect.left + 2, rect.top + 2, self.BLOCK_WIDTH - 10, 2)
        pygame.draw.rect(surface, lighter_color, highlight_rect, border_radius=2)

    def _render_ui(self):
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            game_over_text = self.font_large.render("GAME OVER", True, self.COLOR_WARNING)
            text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 20))
            self.screen.blit(game_over_text, text_rect)
            
            final_score_text = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_UI_TEXT)
            score_rect = final_score_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 + 20))
            self.screen.blit(final_score_text, score_rect)

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # --- Manual Play Example ---
    # This requires a display. Set SDL_VIDEODRIVER to something other than "dummy".
    # For example: os.environ.pop("SDL_VIDEODRIVER", None)
    
    # Check if we can show a display
    can_render = "SDL_VIDEODRIVER" not in os.environ or os.environ["SDL_VIDEODRIVER"] != "dummy"
    
    if not can_render:
        print("Cannot run manual play example in headless mode. Set SDL_VIDEODRIVER to a valid backend.")
    else:
        env = GameEnv()
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Block Stacker")
        clock = pygame.time.Clock()
        
        terminated = False
        truncated = False
        total_reward = 0
        
        while not (terminated or truncated):
            # --- Action Mapping for Human Player ---
            movement = 0 # No-op
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                movement = 4
            
            action = [movement, 0, 0]
            
            # --- Environment Step ---
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    terminated = False
                    truncated = False

            # --- Rendering ---
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(60)

        print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
        env.close()