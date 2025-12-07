
# Generated: 2025-08-28T03:47:20.838856
# Source Brief: brief_05040.md
# Brief Index: 5040

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Arrow keys to move the cursor. Press Space to paint the selected cell, "
        "changing it and all connected cells of the same color to a neighbor's color."
    )

    # User-facing description of the game
    game_description = (
        "A strategic puzzle game. Fill the 10x10 grid with a single color within 25 moves. "
        "Each move spreads a new color like a flood, so plan your choices carefully!"
    )

    # Frames advance only on action
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and grid dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_SIZE = 10
        self.GRID_AREA = self.GRID_SIZE * self.GRID_SIZE
        self.MAX_MOVES = 25

        # Visual constants
        self.COLOR_BG = (26, 28, 44)
        self.COLOR_GRID_LINE = (50, 52, 70)
        self.PALETTE = [
            (59, 179, 199),  # Cyan
            (255, 107, 107), # Red
            (255, 212, 107), # Yellow (for potential expansion)
            (107, 255, 143)  # Green (for potential expansion)
        ]
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_WIN = (173, 255, 47)
        self.COLOR_LOSE = (255, 69, 0)
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 60, bold=True)
        
        # Calculate grid rendering properties
        self.grid_render_size = 360
        self.cell_size = self.grid_render_size // self.GRID_SIZE
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_render_size) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_render_size) // 2
        
        # Initialize state variables to be defined in reset()
        self.grid = None
        self.cursor_pos = None
        self.moves_left = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.particles = None
        self.last_action_was_paint = False
        
        # Initialize state
        self.reset()

        # Validate implementation
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.particles = []
        self.last_action_was_paint = False

        # Create a new random grid with two colors from the palette
        self.grid = self.np_random.integers(0, 2, size=(self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement, space_pressed, _ = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        self.steps += 1
        self.last_action_was_paint = False

        # 1. Handle Movement
        if movement == 1:  # Up
            self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_SIZE) % self.GRID_SIZE
        elif movement == 2:  # Down
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_SIZE
        elif movement == 3:  # Left
            self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_SIZE) % self.GRID_SIZE
        elif movement == 4:  # Right
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_SIZE

        # 2. Handle Paint Action
        if space_pressed:
            self.moves_left -= 1
            self.last_action_was_paint = True
            
            x, y = self.cursor_pos
            original_color = self.grid[y, x]
            
            # Find adjacent cells with different colors
            neighbor_colors = set()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE:
                    neighbor_color = self.grid[ny, nx]
                    if neighbor_color != original_color:
                        neighbor_colors.add(neighbor_color)
            
            pixels_changed = 0
            if neighbor_colors:
                # Choose a random neighbor color to flood with
                target_color = self.np_random.choice(list(neighbor_colors))
                pixels_changed = self._flood_fill(self.cursor_pos, target_color)

            # Calculate continuous reward
            reward += (pixels_changed * 1.0) + ((self.GRID_AREA - pixels_changed) * -0.2)

        # 3. Check Termination Conditions
        terminated = self._check_termination()
        
        if terminated:
            self.game_over = True
            # Check for win condition
            if len(np.unique(self.grid)) == 1:
                reward += 100  # Win bonus
                if pixels_changed > 0: # Check if the last move was the winning one
                    reward += 5
            else: # Must be a loss due to running out of moves
                reward += -100 # Loss penalty
        
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _flood_fill(self, start_pos, target_color):
        start_x, start_y = start_pos
        original_color = self.grid[start_y, start_x]

        if original_color == target_color:
            return 0

        q = deque([start_pos])
        visited = {tuple(start_pos)}
        changed_count = 0

        while q:
            x, y = q.popleft()
            
            if self.grid[y, x] == original_color:
                self.grid[y, x] = target_color
                changed_count += 1
                # sound effect placeholder: # sfx_paint_spread()
                self._spawn_particles(x, y, self.PALETTE[target_color])
                
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and tuple((nx, ny)) not in visited:
                        visited.add(tuple((nx, ny)))
                        q.append([nx, ny])
        
        return changed_count

    def _check_termination(self):
        # Win condition: grid is monochromatic
        if len(np.unique(self.grid)) == 1:
            return True
        # Lose condition: out of moves
        if self.moves_left <= 0:
            return True
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
            "moves_left": self.moves_left,
            "is_win": len(np.unique(self.grid)) == 1 if self.game_over else False
        }

    def _render_game(self):
        # Update and draw particles
        self._update_and_draw_particles()

        # Draw grid cells
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                color_index = self.grid[y, x]
                color = self.PALETTE[color_index]
                rect = pygame.Rect(
                    self.grid_offset_x + x * self.cell_size,
                    self.grid_offset_y + y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, color, rect)

        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_pos = (self.grid_offset_x + i * self.cell_size, self.grid_offset_y)
            end_pos = (self.grid_offset_x + i * self.cell_size, self.grid_offset_y + self.grid_render_size)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, start_pos, end_pos, 2)
            # Horizontal
            start_pos = (self.grid_offset_x, self.grid_offset_y + i * self.cell_size)
            end_pos = (self.grid_offset_x + self.grid_render_size, self.grid_offset_y + i * self.cell_size)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, start_pos, end_pos, 2)

        # Draw cursor with pulsing effect
        cursor_pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2
        cursor_size_bonus = int(cursor_pulse * 4)
        cursor_alpha = int(150 + cursor_pulse * 105)

        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.grid_offset_x + cx * self.cell_size - cursor_size_bonus // 2,
            self.grid_offset_y + cy * self.cell_size - cursor_size_bonus // 2,
            self.cell_size + cursor_size_bonus,
            self.cell_size + cursor_size_bonus
        )
        
        # Create a temporary surface for the transparent cursor
        cursor_surface = pygame.Surface(cursor_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(cursor_surface, (*self.COLOR_CURSOR, cursor_alpha), cursor_surface.get_rect(), border_radius=5)
        pygame.draw.rect(cursor_surface, self.COLOR_CURSOR, cursor_surface.get_rect(), width=2, border_radius=5)
        self.screen.blit(cursor_surface, cursor_rect.topleft)

    def _render_ui(self):
        # Render moves left
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 20))
        
        # Render score
        score_text = self.font_main.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 50))

        # Render game over message
        if self.game_over:
            is_win = len(np.unique(self.grid)) == 1
            message = "YOU WIN!" if is_win else "GAME OVER"
            color = self.COLOR_WIN if is_win else self.COLOR_LOSE
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)
            # sound effect placeholder: # if is_win: sfx_win() else: sfx_lose()

    def _spawn_particles(self, grid_x, grid_y, color):
        center_x = self.grid_offset_x + grid_x * self.cell_size + self.cell_size / 2
        center_y = self.grid_offset_y + grid_y * self.cell_size + self.cell_size / 2
        
        for _ in range(5):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            radius = self.np_random.uniform(3, 6)
            lifespan = self.np_random.integers(20, 40)
            self.particles.append({'pos': [center_x, center_y], 'vel': vel, 'radius': radius, 'color': color, 'lifespan': lifespan, 'max_life': lifespan})

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            
            if p['lifespan'] > 0:
                active_particles.append(p)
                
                # Fade out effect
                alpha = int(255 * (p['lifespan'] / p['max_life']))
                
                # Use gfxdraw for anti-aliased circles
                pos_int = (int(p['pos'][0]), int(p['pos'][1]))
                radius_int = int(p['radius'] * (p['lifespan'] / p['max_life']))
                if radius_int > 0:
                    pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius_int, (*p['color'], alpha))
        
        self.particles = active_particles

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    render_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Pixel Flood")
    clock = pygame.time.Clock()
    
    print("\n" + "="*30)
    print("      Pixel Flood Demo")
    print("="*30)
    print(env.game_description)
    print("\n" + env.user_guide)
    print("\nPress Q to quit.")

    while not done:
        action = [0, 0, 0]  # Default no-op action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
                
                # Map keys to actions for human play
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                
                if event.key == pygame.K_SPACE:
                    action[1] = 1
                
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    action[2] = 1
        
        # If any action was taken, step the environment
        if any(a != 0 for a in action):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step: {info['steps']}, Moves Left: {info['moves_left']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")
            if terminated:
                print("Game Over! Final Score:", info['score'])
        else:
            # For turn-based games, we still need to get the observation to render
            obs = env._get_observation()

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS for human play

    env.close()