import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:47:19.584084
# Source Brief: brief_03007.md
# Brief Index: 3007
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Stack falling blocks and match three of the same color vertically to score points and create gold blocks."
    )
    user_guide = (
        "Controls: Use ← and → to move the falling block, ↓ to speed up its descent, and space to instantly drop it."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GAME_DURATION_SECONDS = 120
        self.MAX_STEPS = self.GAME_DURATION_SECONDS * self.FPS

        # Playfield Grid
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 15
        self.BLOCK_SIZE = 20
        self.PLAYFIELD_WIDTH = self.GRID_WIDTH * self.BLOCK_SIZE
        self.PLAYFIELD_HEIGHT = self.GRID_HEIGHT * self.BLOCK_SIZE
        self.PLAYFIELD_X_OFFSET = (self.WIDTH - self.PLAYFIELD_WIDTH) // 2
        self.PLAYFIELD_Y_OFFSET = self.HEIGHT - self.PLAYFIELD_HEIGHT - 20

        # Colors
        self.COLOR_BG = (25, 25, 40)
        self.COLOR_GRID = (45, 45, 60)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TEXT_SHADOW = (10, 10, 20)
        self.BLOCK_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 120, 255),  # Blue
        ]
        self.GOLD_BLOCK_COLOR = (255, 215, 0)
        self.GOLD_BLOCK_ID = 3

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 22)

        # --- Game State (initialized in reset) ---
        self.steps = None
        self.score = None
        self.game_over = None
        self.timer = None
        self.playfield = None
        self.falling_block = None
        self.next_block = None
        self.particles = None
        self.base_fall_speed = None
        self.current_fall_speed = None
        self.last_move_time = None
        self.move_cooldown = 5 # steps

        self.reset()
        # self.validate_implementation() # Commented out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_STEPS
        
        # Difficulty scaling setup
        self.base_fall_speed = 0.75
        self.current_fall_speed = self.base_fall_speed

        # Initialize playfield grid (stores color indices or None)
        self.playfield = [[None for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
        self.particles = []
        self.last_move_time = 0

        # Create initial blocks
        self.next_block = self._create_block()
        self._spawn_new_block()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.timer -= 1
        reward = 0

        # Unpack factorized action
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1

        # --- Game Logic ---
        self._handle_input(movement, space_pressed)
        self._update_difficulty()

        landed = self._update_falling_block(movement)

        if landed:
            # Place block, check for matches/collapses, spawn next block
            placement_reward, match_reward, collapse_penalty, topped_out = self._process_landing()
            reward += placement_reward + match_reward + collapse_penalty
            
            if topped_out:
                self.game_over = True
                reward = -100.0 # Severe penalty for topping out
        
        self._update_particles()
        
        # --- Termination ---
        terminated = self._check_termination()
        if terminated and not self.game_over:
            if self.score >= 100:
                reward += 100.0 # Victory bonus
            self.game_over = True

        truncated = False
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed):
        if not self.falling_block: return

        # Horizontal movement with cooldown for tap vs hold
        if self.steps > self.last_move_time + self.move_cooldown:
            if movement == 3:  # Left
                new_x = max(0, self.falling_block['x'] - 1)
                if self._is_valid_position(new_x, self.falling_block['y_grid']):
                    self.falling_block['x'] = new_x
                    self.last_move_time = self.steps
            elif movement == 4:  # Right
                new_x = min(self.GRID_WIDTH - 1, self.falling_block['x'] + 1)
                if self._is_valid_position(new_x, self.falling_block['y_grid']):
                    self.falling_block['x'] = new_x
                    self.last_move_time = self.steps
        
        # Instant drop
        if space_pressed:
            # Find landing spot
            y = self.falling_block['y']
            while self._is_valid_position(self.falling_block['x'], math.floor((y + self.BLOCK_SIZE) / self.BLOCK_SIZE)):
                y += self.BLOCK_SIZE
            self.falling_block['y'] = y - self.BLOCK_SIZE
            # Play sound effect placeholder
            # sfx: instant_drop

    def _update_difficulty(self):
        # Increase fall speed every 30 seconds
        if self.steps > 0 and self.steps % (30 * self.FPS) == 0:
            self.base_fall_speed += 0.2
            self.current_fall_speed = self.base_fall_speed

    def _update_falling_block(self, movement):
        if not self.falling_block: return False

        # Apply fall speed (accelerated if down is pressed)
        speed_multiplier = 3.0 if movement == 2 else 1.0
        self.falling_block['y'] += self.current_fall_speed * speed_multiplier
        
        # Check for collision
        next_y_grid = math.floor((self.falling_block['y'] + self.BLOCK_SIZE) / self.BLOCK_SIZE)
        
        if not self._is_valid_position(self.falling_block['x'], next_y_grid):
            return True # Landed
        
        self.falling_block['y_grid'] = math.floor(self.falling_block['y'] / self.BLOCK_SIZE)
        return False

    def _process_landing(self):
        if not self.falling_block: return 0, 0, 0, False

        # 1. Place the block in the grid
        final_x = self.falling_block['x']
        final_y_grid = self.falling_block['y_grid']
        
        # Ensure placement is within bounds (can happen with instant drop)
        final_y_grid = min(self.GRID_HEIGHT - 1, final_y_grid)

        self.playfield[final_y_grid][final_x] = self.falling_block['color_idx']
        placement_reward = 0.1
        # Play sound effect placeholder
        # sfx: block_land

        # 2. Check for vertical matches
        match_reward = self._check_matches(final_x, final_y_grid)

        # 3. Check for collapses
        collapse_penalty = self._handle_collapses()

        # 4. Spawn new block and check for top-out
        topped_out = self._spawn_new_block()

        return placement_reward, match_reward, collapse_penalty, topped_out

    def _check_matches(self, x, y):
        # Check for 3 vertical blocks of the same color
        if y < 2: return 0 # Not enough space above for a match
        
        color_idx = self.playfield[y][x]
        if color_idx is None or color_idx == self.GOLD_BLOCK_ID: return 0

        if (self.playfield[y-1][x] == color_idx and self.playfield[y-2][x] == color_idx):
            # Match found!
            # Play sound effect placeholder
            # sfx: match_success
            self.score += 5
            
            # Create particles at each matched block's position
            for i in range(3):
                self._create_particles(x, y - i, self.BLOCK_COLORS[color_idx])

            # Remove matched blocks and place a gold block at the bottom
            self.playfield[y][x] = self.GOLD_BLOCK_ID
            self.playfield[y-1][x] = None
            self.playfield[y-2][x] = None
            return 5.0
            
        return 0

    def _handle_collapses(self):
        # Iteratively let unsupported blocks fall down
        total_penalty = 0
        made_change = True
        while made_change:
            made_change = False
            for y in range(self.GRID_HEIGHT - 2, -1, -1): # From second to last row upwards
                for x in range(self.GRID_WIDTH):
                    if self.playfield[y][x] is not None and self.playfield[y+1][x] is None:
                        # This block is unsupported, move it down
                        self.playfield[y+1][x] = self.playfield[y][x]
                        self.playfield[y][x] = None
                        made_change = True
        
        # Check for blocks that fell off the bottom (not possible with this logic, but good practice)
        # In this design, collapses just mean rearrangement, not loss of blocks.
        # The brief implies loss, so let's modify. A block is "unsupported" if there is nothing under it all the way down.
        
        lost_blocks_count = 0
        for x in range(self.GRID_WIDTH):
            # Find the first solid block from the bottom up
            support_y = self.GRID_HEIGHT
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.playfield[y][x] is not None:
                    support_y = y
                    break
            
            # Any block above an empty space below the support is lost
            for y in range(support_y - 1, -1, -1):
                if self.playfield[y][x] is not None:
                    # Play sound effect placeholder
                    # sfx: collapse
                    self._create_particles(x, y, (150,150,150), 3) # Grey failure particles
                    self.playfield[y][x] = None
                    lost_blocks_count += 1

        if lost_blocks_count > 0:
            collapse_penalty = -10.0 * lost_blocks_count
            self.score -= 10 * lost_blocks_count
            return collapse_penalty
        
        return 0

    def _spawn_new_block(self):
        self.falling_block = self.next_block
        self.next_block = self._create_block()

        # Check for game over (top out)
        if not self._is_valid_position(self.falling_block['x'], self.falling_block['y_grid']):
            self.falling_block = None # Can't place it
            return True # Topped out
        return False

    def _create_block(self):
        color_idx = self.np_random.integers(0, len(self.BLOCK_COLORS))
        return {
            'x': self.np_random.integers(0, self.GRID_WIDTH),
            'y': 0.0, # Pixel position for smooth falling
            'y_grid': 0, # Grid position for collision
            'color_idx': color_idx
        }

    def _is_valid_position(self, x, y_grid):
        # Check boundaries
        if not (0 <= x < self.GRID_WIDTH and 0 <= y_grid < self.GRID_HEIGHT):
            return False
        # Check collision with existing blocks
        if self.playfield[y_grid][x] is not None:
            return False
        return True

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # Gravity
            p['life'] -= 1

    def _create_particles(self, grid_x, grid_y, color, count=10):
        px = self.PLAYFIELD_X_OFFSET + grid_x * self.BLOCK_SIZE + self.BLOCK_SIZE // 2
        py = self.PLAYFIELD_Y_OFFSET + grid_y * self.BLOCK_SIZE + self.BLOCK_SIZE // 2
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                'x': px, 'y': py,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': self.np_random.integers(15, 30),
                'color': color
            })

    def _check_termination(self):
        return self.score >= 100 or self.timer <= 0 or self.steps >= self.MAX_STEPS

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_playfield_grid()
        self._draw_stacked_blocks()
        self._draw_falling_block()
        self._draw_particles()

    def _draw_playfield_grid(self):
        for x in range(self.GRID_WIDTH + 1):
            start_pos = (self.PLAYFIELD_X_OFFSET + x * self.BLOCK_SIZE, self.PLAYFIELD_Y_OFFSET)
            end_pos = (self.PLAYFIELD_X_OFFSET + x * self.BLOCK_SIZE, self.PLAYFIELD_Y_OFFSET + self.PLAYFIELD_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)
        for y in range(self.GRID_HEIGHT + 1):
            start_pos = (self.PLAYFIELD_X_OFFSET, self.PLAYFIELD_Y_OFFSET + y * self.BLOCK_SIZE)
            end_pos = (self.PLAYFIELD_X_OFFSET + self.PLAYFIELD_WIDTH, self.PLAYFIELD_Y_OFFSET + y * self.BLOCK_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)

    def _draw_stacked_blocks(self):
        for y, row in enumerate(self.playfield):
            for x, color_idx in enumerate(row):
                if color_idx is not None:
                    color = self.GOLD_BLOCK_COLOR if color_idx == self.GOLD_BLOCK_ID else self.BLOCK_COLORS[color_idx]
                    self._draw_block(
                        self.screen,
                        self.PLAYFIELD_X_OFFSET + x * self.BLOCK_SIZE,
                        self.PLAYFIELD_Y_OFFSET + y * self.BLOCK_SIZE,
                        color, self.BLOCK_SIZE
                    )

    def _draw_falling_block(self):
        if self.falling_block:
            color = self.BLOCK_COLORS[self.falling_block['color_idx']]
            self._draw_block(
                self.screen,
                self.PLAYFIELD_X_OFFSET + self.falling_block['x'] * self.BLOCK_SIZE,
                self.PLAYFIELD_Y_OFFSET + self.falling_block['y'],
                color, self.BLOCK_SIZE
            )

    def _draw_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 30.0))))
            color_with_alpha = p['color'] + (alpha,)
            size = max(1, int(p['life']/10))
            
            # Create a temporary surface for alpha blending
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color_with_alpha, (0, 0, size, size))
            self.screen.blit(temp_surf, (int(p['x']) - size//2, int(p['y']) - size//2))


    def _draw_block(self, surface, x, y, color, size):
        x, y = int(x), int(y)
        # 3D-effect by drawing borders
        light_color = tuple(min(255, c + 40) for c in color)
        dark_color = tuple(max(0, c - 40) for c in color)
        
        pygame.draw.rect(surface, dark_color, (x, y, size, size))
        pygame.draw.rect(surface, color, (x, y, size - 2, size - 2))
        pygame.draw.rect(surface, light_color, (x, y, size - 2, 1)) # Top highlight
        pygame.draw.rect(surface, light_color, (x, y, 1, size - 2)) # Left highlight


    def _render_ui(self):
        # Score
        self._draw_text(f"SCORE: {self.score}", self.font_large, (40, 30))
        
        # Timer
        time_left_sec = self.timer // self.FPS
        timer_text = f"TIME: {time_left_sec}"
        timer_color = (255, 100, 100) if time_left_sec < 10 else self.COLOR_TEXT
        self._draw_text(timer_text, self.font_large, (self.WIDTH - 150, 30), color=timer_color)

        # Next Block Preview
        self._draw_text("NEXT", self.font_medium, (self.WIDTH - 115, self.HEIGHT - 100))
        if self.next_block:
            color = self.BLOCK_COLORS[self.next_block['color_idx']]
            self._draw_block(self.screen, self.WIDTH - 120, self.HEIGHT - 70, color, self.BLOCK_SIZE * 1.5)
            
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0,0))
            
            end_text = "YOU WIN!" if self.score >= 100 else "GAME OVER"
            self._draw_text(end_text, pygame.font.Font(None, 72), (self.WIDTH // 2, self.HEIGHT // 2 - 20), centered=True)

    def _draw_text(self, text, font, pos, color=None, centered=False):
        if color is None: color = self.COLOR_TEXT
        shadow_surface = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_surface = font.render(text, True, color)
        
        x, y = pos
        if centered:
            text_rect = text_surface.get_rect(center=(x, y))
            shadow_rect = shadow_surface.get_rect(center=(x + 2, y + 2))
        else:
            text_rect = text_surface.get_rect(topleft=(x, y))
            shadow_rect = shadow_surface.get_rect(topleft=(x + 2, y + 2))

        self.screen.blit(shadow_surface, shadow_rect)
        self.screen.blit(text_surface, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Unset the dummy video driver to allow for display
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.display.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Stacker")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        # --- Manual Control ---
        movement = 0 # none
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        if not terminated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            elif keys[pygame.K_DOWN]: movement = 2
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
            
            if reward != 0:
                print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

        # --- Rendering ---
        # The observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.FPS)
        
    env.close()