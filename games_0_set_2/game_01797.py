# Generated: 2025-08-28T02:44:47.006191
# Source Brief: brief_01797.md
# Brief Index: 1797

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Hold Space to activate the paddle and deflect falling blocks."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist block-breaking puzzle. Position your paddle and activate it at the right moment to clear falling blocks. Clear 50 blocks to win, but let one reach the bottom and you lose."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.CELL_WIDTH = self.WIDTH // self.GRID_WIDTH
        self.CELL_HEIGHT = self.HEIGHT // self.GRID_HEIGHT
        
        # Colors
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_PADDLE_ACTIVE = (0, 255, 255)
        self.COLOR_PADDLE_GLOW = (0, 150, 150, 100)
        self.BLOCK_COLORS = [
            (255, 87, 34),   # Deep Orange
            (3, 169, 244),   # Light Blue
            (255, 235, 59),  # Yellow
            (139, 195, 74),  # Light Green
            (233, 30, 99),   # Pink
        ]
        self.COLOR_TEXT = (240, 240, 240)
        
        # Game parameters
        self.MAX_STEPS = 1500  # Extended to allow for more gameplay
        self.WIN_CONDITION = 50
        self.INITIAL_FALL_SPEED = 1.5
        self.SPEED_INCREASE_INTERVAL = 10
        self.SPEED_INCREASE_AMOUNT = 0.2
        self.PADDLE_SPEED = 1 # grid units per action
        self.PADDLE_HEIGHT = self.CELL_HEIGHT
        self.BLOCK_SPAWN_RATE_INITIAL = 90 # frames
        self.BLOCK_SPAWN_RATE_MIN = 30

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- State Variables ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.np_random = None
        
        self.paddle_pos = 0
        self.blocks = []
        self.particles = []
        self.blocks_cleared = 0
        self.current_fall_speed = 0.0
        self.time_to_next_block = 0
        
        # Initialize state by calling reset
        # self.reset() is called by the wrapper usually, so we don't call it here
        # to avoid double-initialization. However, for validation, we need a state.
        self.np_random = np.random.default_rng() # for validation
        self.paddle_pos = self.GRID_WIDTH // 2

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        
        self.paddle_pos = self.GRID_WIDTH // 2
        self.blocks = []
        self.particles = []
        self.blocks_cleared = 0
        self.current_fall_speed = self.INITIAL_FALL_SPEED
        self.time_to_next_block = self.BLOCK_SPAWN_RATE_INITIAL // 2
        
        return self._get_observation(space_held=False), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing but return the final state
            space_held = action[1] == 1
            return self._get_observation(space_held), 0.0, True, False, self._get_info()

        self.steps += 1
        
        movement = action[0]
        space_held = action[1] == 1
        
        reward, terminated = self._update_game_state(movement, space_held)
        
        self.score += reward
        self.game_over = terminated
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True
            
        return (
            self._get_observation(space_held),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_game_state(self, movement, space_held):
        reward = 0.0
        terminated = False
        blocks_broken_this_step = 0

        # 1. Handle player input and paddle movement
        if movement == 3:  # Left
            self.paddle_pos = max(0, self.paddle_pos - self.PADDLE_SPEED)
        elif movement == 4:  # Right
            self.paddle_pos = min(self.GRID_WIDTH - 1, self.paddle_pos + self.PADDLE_SPEED)
        elif movement == 0: # No-op
            reward -= 0.02

        # 2. Update blocks
        paddle_rect = pygame.Rect(
            self.paddle_pos * self.CELL_WIDTH, 
            self.HEIGHT - self.PADDLE_HEIGHT, 
            self.CELL_WIDTH, 
            self.PADDLE_HEIGHT
        )
        
        for block in reversed(self.blocks):
            block['pos'][1] += self.current_fall_speed
            
            block_rect = pygame.Rect(
                block['pos'][0], block['pos'][1], self.CELL_WIDTH, self.CELL_HEIGHT
            )

            # Check for paddle collision
            activation_zone = paddle_rect.copy()
            activation_zone.y -= self.CELL_HEIGHT / 2 # Generous activation zone
            
            if space_held and activation_zone.colliderect(block_rect):
                # sfx: block_break.wav
                self._create_particles(block_rect.center, block['color'])
                self.blocks.remove(block)
                self.blocks_cleared += 1
                blocks_broken_this_step += 1
                reward += 1.0  # +1 for breaking a block
                reward += 0.1  # +0.1 for deflecting
                
                # Check for difficulty increase
                if self.blocks_cleared > 0 and self.blocks_cleared % self.SPEED_INCREASE_INTERVAL == 0:
                    self.current_fall_speed += self.SPEED_INCREASE_AMOUNT
                    # sfx: level_up.wav

            # Check for game over
            elif block_rect.bottom >= self.HEIGHT:
                # sfx: game_over.wav
                terminated = True
                reward -= 100.0
                self.blocks.remove(block)

        # Multi-break reward
        if blocks_broken_this_step > 1:
            reward += 2.0

        # 3. Update particles
        for p in reversed(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

        # 4. Spawn new blocks
        self.time_to_next_block -= 1
        if self.time_to_next_block <= 0:
            self._spawn_block()
            spawn_rate = self.BLOCK_SPAWN_RATE_INITIAL - (self.blocks_cleared // 2)
            self.time_to_next_block = max(self.BLOCK_SPAWN_RATE_MIN, spawn_rate)

        # 5. Check for win condition
        if self.blocks_cleared >= self.WIN_CONDITION:
            # sfx: win_game.wav
            self.win = True
            terminated = True
            reward += 100.0

        return reward, terminated

    def _spawn_block(self):
        x_pos = self.np_random.integers(0, self.GRID_WIDTH) * self.CELL_WIDTH
        color_idx = self.np_random.integers(0, len(self.BLOCK_COLORS))
        
        new_block = {
            'pos': [x_pos, -self.CELL_HEIGHT],
            'color': self.BLOCK_COLORS[color_idx]
        }
        self.blocks.append(new_block)

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _get_observation(self, space_held):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_grid()
        self._render_particles()
        self._render_blocks()
        self._render_paddle(space_held)
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for x in range(0, self.WIDTH, self.CELL_WIDTH):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_HEIGHT):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_paddle(self, active):
        paddle_rect = pygame.Rect(
            self.paddle_pos * self.CELL_WIDTH, 
            self.HEIGHT - self.PADDLE_HEIGHT, 
            self.CELL_WIDTH, 
            self.PADDLE_HEIGHT
        )

        if active:
            # Draw glow
            glow_center = paddle_rect.center
            glow_radius = int(self.CELL_WIDTH * 0.75)
            # Create a temporary surface for the glow
            glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            # FIX: filled_circle takes 5 arguments: surface, x, y, radius, color
            pygame.gfxdraw.filled_circle(glow_surface, glow_radius, glow_radius, glow_radius, self.COLOR_PADDLE_GLOW)
            self.screen.blit(glow_surface, (glow_center[0] - glow_radius, glow_center[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
            
            # Draw active paddle
            pygame.draw.rect(self.screen, self.COLOR_PADDLE_ACTIVE, paddle_rect, border_radius=3)
        else:
            pygame.draw.rect(self.screen, self.COLOR_PADDLE, paddle_rect, border_radius=3)
    
    def _render_blocks(self):
        for block in self.blocks:
            rect = pygame.Rect(
                int(block['pos'][0]), int(block['pos'][1]), self.CELL_WIDTH, self.CELL_HEIGHT
            )
            pygame.draw.rect(self.screen, block['color'], rect, border_radius=3)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, 255 * (p['lifespan'] / 30))
            color_with_alpha = p['color'] + (alpha,)
            size = max(1, int(4 * (p['lifespan'] / 30)))
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), size)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Blocks Cleared
        cleared_text = self.font_main.render(f"Cleared: {self.blocks_cleared}/{self.WIN_CONDITION}", True, self.COLOR_TEXT)
        cleared_rect = cleared_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(cleared_text, cleared_rect)

        # Game Over / Win message
        if self.game_over:
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            
            end_text = self.font_game_over.render(message, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            
            # Draw a semi-transparent background for readability
            bg_surf = pygame.Surface(end_rect.size, pygame.SRCALPHA)
            bg_surf.fill((0, 0, 0, 150))
            self.screen.blit(bg_surf, end_rect.topleft)
            
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "blocks_cleared": self.blocks_cleared,
        }
    
    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game with keyboard controls
    # It requires a display, so we unset the dummy driver
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Use a separate display for human play
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Deflector")
    clock = pygame.time.Clock()

    action = np.array([0, 0, 0]) # No-op, no space, no shift

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                terminated = False

        if not terminated:
            keys = pygame.key.get_pressed()
            
            # Action mapping
            movement = 0 # none
            if keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 0 # Not used in this game
            
            action = np.array([movement, space_held, shift_held])
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if reward != 0:
                print(f"Step: {info['steps']}, Score: {info['score']:.2f}, Reward: {reward:.2f}, Cleared: {info['blocks_cleared']}")
            
            if terminated:
                print(f"--- Episode Finished ---")
                print(f"Final Score: {info['score']:.2f}, Steps: {info['steps']}")

        # Render the observation to the display
        # The observation is (H, W, C), but pygame blit needs a surface
        # And surfarray.make_surface needs (W, H, C)
        obs_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(obs_surface, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Limit to 30 FPS for human play

    env.close()