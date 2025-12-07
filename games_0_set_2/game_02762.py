
# Generated: 2025-08-27T21:21:40.863403
# Source Brief: brief_02762.md
# Brief Index: 2762

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to paint the selected square. "
        "Press Shift to cycle through the color palette."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A creative puzzle game where you recreate a target pixel art image. "
        "Select colors and paint squares on your canvas to match the target image as closely as possible before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 600
    GRID_DIM = 10

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_TEXT = (230, 230, 240)
    COLOR_TEXT_SHADOW = (10, 10, 15)
    COLOR_GRID_BG = (40, 50, 70)
    COLOR_GRID_LINE = (60, 70, 90)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_WIN = (100, 255, 150, 200)
    COLOR_LOSE = (255, 100, 100, 200)

    # Game Palette (Inspired by PICO-8)
    COLOR_PALETTE = [
        (0, 0, 0),        # Black
        (29, 43, 83),     # Dark Blue
        (126, 37, 83),    # Dark Purple
        (0, 135, 81),     # Dark Green
        (171, 82, 54),    # Brown
        (95, 87, 79),     # Dark Grey
        (194, 195, 199),  # Light Grey
        (255, 241, 232),  # White
        (255, 0, 77),     # Red
        (255, 163, 0),    # Orange
        (255, 236, 39),   # Yellow
        (0, 228, 54),     # Green
        (41, 173, 255),   # Blue
        (131, 118, 156),  # Indigo
        (255, 119, 168),  # Pink
        (255, 204, 170),  # Peach
    ]
    
    # We'll use a subset of the palette for each game
    PALETTE_SIZE = 8
    
    # Layout
    GRID_CELL_SIZE = 24
    GRID_WIDTH = GRID_HEIGHT = GRID_DIM * GRID_CELL_SIZE
    TARGET_GRID_POS = (SCREEN_WIDTH // 2 - GRID_WIDTH - 20, 100)
    PLAYER_GRID_POS = (SCREEN_WIDTH // 2 + 20, 100)
    PALETTE_CELL_SIZE = 30


    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Etc...
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.timer = self.MAX_STEPS
        self.game_over = False
        self.win = False

        self.cursor_pos = np.array([0, 0])
        
        # Select a random subset of the main palette for this round
        palette_indices = self.np_random.choice(len(self.COLOR_PALETTE), self.PALETTE_SIZE, replace=False)
        self.active_palette_indices = palette_indices
        self.active_palette = [self.COLOR_PALETTE[i] for i in palette_indices]
        
        # The neutral color is the first in the active palette
        self.neutral_color_idx = 0
        self.selected_color_idx = 0

        self.target_image = self.np_random.integers(0, self.PALETTE_SIZE, size=(self.GRID_DIM, self.GRID_DIM))
        self.player_canvas = np.full((self.GRID_DIM, self.GRID_DIM), self.neutral_color_idx, dtype=int)
        
        self.score = self._calculate_match_percentage()

        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.reward_50_given = False
        self.reward_75_given = False
        
        self.particles = []

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        # Update game logic
        reward = 0
        self.steps += 1
        if not self.game_over:
            self.timer -= 1

        # --- Handle Input ---
        # 1. Cursor Movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos = np.clip(self.cursor_pos, 0, self.GRID_DIM - 1)

        # 2. Color Cycle (on press)
        if shift_held and not self.prev_shift_held:
            self.selected_color_idx = (self.selected_color_idx + 1) % self.PALETTE_SIZE
            # sfx: color_cycle.wav

        # 3. Paint (on press)
        if space_held and not self.prev_space_held:
            cx, cy = self.cursor_pos
            current_color = self.player_canvas[cy, cx]
            target_color = self.target_image[cy, cx]
            selected_color = self.selected_color_idx

            if current_color != selected_color:
                was_correct = current_color == target_color
                is_now_correct = selected_color == target_color
                
                if not was_correct and is_now_correct:
                    reward += 0.1  # Fixed a wrong pixel
                elif was_correct and not is_now_correct:
                    reward -= 0.1  # Ruined a correct pixel (higher penalty)
                elif not was_correct and not is_now_correct:
                    reward -= 0.01 # Wasted paint
                
                self.player_canvas[cy, cx] = selected_color
                self._add_particles(cx, cy)
                # sfx: paint.wav
        
        self.prev_shift_held = shift_held
        self.prev_space_held = space_held

        # --- Update Game State ---
        self._update_particles()
        self.score = self._calculate_match_percentage()

        # Milestone rewards
        if self.score >= 50 and not self.reward_50_given:
            reward += 5; self.reward_50_given = True # sfx: milestone.wav
        if self.score >= 75 and not self.reward_75_given:
            reward += 10; self.reward_75_given = True # sfx: milestone_major.wav
            
        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.win:
                reward += 100 # sfx: win.wav
            else:
                reward -= 50 # sfx: lose.wav
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _calculate_match_percentage(self):
        correct_pixels = np.sum(self.player_canvas == self.target_image)
        total_pixels = self.GRID_DIM * self.GRID_DIM
        return (correct_pixels / total_pixels) * 100.0

    def _check_termination(self):
        if self.score >= 90.0:
            self.win = True
            return True
        if self.timer <= 0:
            self.win = False
            return True
        return False
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _add_particles(self, grid_x, grid_y):
        px, py = self.PLAYER_GRID_POS
        center_x = px + grid_x * self.GRID_CELL_SIZE + self.GRID_CELL_SIZE / 2
        center_y = py + grid_y * self.GRID_CELL_SIZE + self.GRID_CELL_SIZE / 2
        color = self.active_palette[self.player_canvas[grid_y, grid_x]]
        
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(10, 20)
            self.particles.append({'pos': [center_x, center_y], 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _render_text(self, text, font, color, center_pos, shadow=True):
        if shadow:
            text_surf_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            text_rect_shadow = text_surf_shadow.get_rect(center=(center_pos[0] + 2, center_pos[1] + 2))
            self.screen.blit(text_surf_shadow, text_rect_shadow)
        
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect(center=center_pos)
        self.screen.blit(text_surf, text_rect)

    def _draw_grid(self, surface, pos, grid_data, palette):
        px, py = pos
        pygame.draw.rect(surface, self.COLOR_GRID_BG, (px, py, self.GRID_WIDTH, self.GRID_HEIGHT))
        for y in range(self.GRID_DIM):
            for x in range(self.GRID_DIM):
                color_index = grid_data[y, x]
                color = palette[color_index]
                rect = (px + x * self.GRID_CELL_SIZE, py + y * self.GRID_CELL_SIZE, self.GRID_CELL_SIZE, self.GRID_CELL_SIZE)
                pygame.draw.rect(surface, color, rect)
        
        for i in range(self.GRID_DIM + 1):
            pygame.draw.line(surface, self.COLOR_GRID_LINE, (px, py + i * self.GRID_CELL_SIZE), (px + self.GRID_WIDTH, py + i * self.GRID_CELL_SIZE))
            pygame.draw.line(surface, self.COLOR_GRID_LINE, (px + i * self.GRID_CELL_SIZE, py), (px + i * self.GRID_CELL_SIZE, py + self.GRID_HEIGHT))

    def _render_game(self):
        self._render_text("Target", self.font_medium, self.COLOR_TEXT, (self.TARGET_GRID_POS[0] + self.GRID_WIDTH // 2, self.TARGET_GRID_POS[1] - 20))
        self._draw_grid(self.screen, self.TARGET_GRID_POS, self.target_image, self.active_palette)

        self._render_text("Your Canvas", self.font_medium, self.COLOR_TEXT, (self.PLAYER_GRID_POS[0] + self.GRID_WIDTH // 2, self.PLAYER_GRID_POS[1] - 20))
        self._draw_grid(self.screen, self.PLAYER_GRID_POS, self.player_canvas, self.active_palette)
        
        for p in self.particles:
            size = max(0, p['lifespan'] / 5)
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(size))

        if not self.game_over:
            cursor_x = self.PLAYER_GRID_POS[0] + self.cursor_pos[0] * self.GRID_CELL_SIZE
            cursor_y = self.PLAYER_GRID_POS[1] + self.cursor_pos[1] * self.GRID_CELL_SIZE
            
            pulse = (math.sin(self.steps * 0.3) + 1) / 2
            thickness = int(2 + pulse * 2)
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, (cursor_x, cursor_y, self.GRID_CELL_SIZE, self.GRID_CELL_SIZE), thickness)

    def _render_ui(self):
        self._render_text(f"Match: {self.score:.1f}%", self.font_medium, self.COLOR_TEXT, (100, 30))
        
        timer_bar_width = 200
        timer_ratio = max(0, self.timer / self.MAX_STEPS)
        current_bar_width = int(timer_bar_width * timer_ratio)
        bar_color = (255, 255, 0) if timer_ratio > 0.5 else (255, 165, 0) if timer_ratio > 0.2 else (255, 0, 0)
        
        self._render_text("Time", self.font_medium, self.COLOR_TEXT, (self.SCREEN_WIDTH - 100 - timer_bar_width/2, 30))
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (self.SCREEN_WIDTH - 220, 50, timer_bar_width + 4, 24))
        pygame.draw.rect(self.screen, bar_color, (self.SCREEN_WIDTH - 218, 52, current_bar_width, 20))

        palette_width = self.PALETTE_SIZE * (self.PALETTE_CELL_SIZE + 5) - 5
        palette_start_x = (self.SCREEN_WIDTH - palette_width) // 2
        
        self._render_text("Color Palette", self.font_small, self.COLOR_TEXT, (self.SCREEN_WIDTH // 2, 360))
        
        for i, color in enumerate(self.active_palette):
            x = palette_start_x + i * (self.PALETTE_CELL_SIZE + 5)
            y = 375
            pygame.draw.rect(self.screen, color, (x, y, self.PALETTE_CELL_SIZE, self.PALETTE_CELL_SIZE))
            if i == self.selected_color_idx:
                pygame.draw.rect(self.screen, self.COLOR_CURSOR, (x-2, y-2, self.PALETTE_CELL_SIZE+4, self.PALETTE_CELL_SIZE+4), 3)

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        message = "VICTORY!" if self.win else "TIME'S UP"
        color = self.COLOR_WIN if self.win else self.COLOR_LOSE
        
        overlay.fill(color)
        self.screen.blit(overlay, (0, 0))
        
        self._render_text(message, self.font_large, self.COLOR_TEXT, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 20))
        self._render_text(f"Final Match: {self.score:.1f}%", self.font_medium, self.COLOR_TEXT, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 + 30))

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    # Game loop
    running = True
    is_game_over_state = False
    
    # Create a display for interactive testing
    pygame.display.set_caption("Pixel Painter")
    display_surf = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    while running:
        # Pygame event handling
        pygame_action = [0, 0, 0] # [movement, space, shift]
        
        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Handle reset on key press after game over
            if event.type == pygame.KEYDOWN and is_game_over_state:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    is_game_over_state = False

        # If game is not over, process player input
        if not is_game_over_state:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: pygame_action[0] = 1
            elif keys[pygame.K_DOWN]: pygame_action[0] = 2
            elif keys[pygame.K_LEFT]: pygame_action[0] = 3
            elif keys[pygame.K_RIGHT]: pygame_action[0] = 4
            
            if keys[pygame.K_SPACE]: pygame_action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: pygame_action[2] = 1

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(pygame_action)
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.1f}%")

            if terminated:
                is_game_over_state = True
                print("--- GAME OVER ---")
                print(f"Final Score: {info['score']:.1f}% | Press 'R' to restart.")


        # Render the observation to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_surf.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Run at 30 FPS for smooth manual play

    pygame.quit()