import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to match the "
        "highlighted group of fruits. Hold Shift to get a hint."
    )

    game_description = (
        "Match cascading fruits in this vibrant puzzle game. Form groups of 3 or more "
        "to score points and trigger combos. Reach 5000 points before you run out of moves!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_ROWS, self.GRID_COLS = 8, 8
        self.CELL_SIZE = 42
        self.GRID_WIDTH = self.GRID_COLS * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_ROWS * self.CELL_SIZE
        self.GRID_X = (self.SCREEN_WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y = self.SCREEN_HEIGHT - self.GRID_HEIGHT - 10
        self.NUM_FRUIT_TYPES = 5
        self.MAX_MOVES = 100
        self.WIN_SCORE = 5000
        self.MAX_STEPS = 2000

        # Colors
        self.COLOR_BG = (25, 20, 40)
        self.COLOR_GRID = (40, 35, 60)
        self.COLOR_UI_TEXT = (240, 240, 255)
        self.COLOR_UI_BG = (50, 45, 70, 180)
        self.FRUIT_COLORS = [
            (220, 50, 50),   # Red
            (240, 220, 60),  # Yellow
            (100, 200, 80),  # Green
            (180, 80, 230),  # Purple
            (250, 150, 50),  # Orange
        ]
        self.PARTICLE_COLORS = [c for c in self.FRUIT_COLORS]

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
        self.font_ui = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Arial", 48, bold=True)
        self.font_combo = pygame.font.SysFont("Arial", 24, bold=True)

        # Game state variables are initialized in reset()
        self.grid = None
        self.visual_grid = None
        self.cursor_pos = None
        self.score = None
        self.moves_left = None
        self.game_over = None
        self.steps = None
        self.game_state = None
        self.animation_timer = None
        self.particles = None
        self.combo_multiplier = None
        self.matched_cells_for_anim = None
        self.fall_map = None
        self.space_was_held = None
        self.shift_was_held = None
        self.hint_cells = None
        self.hint_timer = None
        self.last_match_reward = None
        self.floating_texts = None

        # self.reset() is called by the wrapper or user, not in __init__
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.combo_multiplier = 1
        self.last_match_reward = 0
        
        self.game_state = 'IDLE' # IDLE, MATCH, FALL, REFILL, GAMEOVER
        self.animation_timer = 0
        self.particles = []
        self.floating_texts = []
        self.matched_cells_for_anim = set()
        self.fall_map = {}
        
        self.space_was_held = False
        self.shift_was_held = False
        self.hint_cells = set()
        self.hint_timer = 0

        self._populate_grid()
        self._create_visual_grid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small penalty for time passing
        terminated = False
        
        self._update_animations()

        if self.game_state == 'IDLE':
            reward += self._handle_input(action)
        elif self.game_state == 'MATCH' and self.animation_timer <= 0:
            self._start_fall()
        elif self.game_state == 'FALL' and self.animation_timer <= 0:
            self._apply_fall()
            self._start_refill()
        elif self.game_state == 'REFILL' and self.animation_timer <= 0:
            reward += self._check_for_chains()

        self.steps += 1
        
        if self.game_state == 'IDLE' and (self.moves_left <= 0 or self.score >= self.WIN_SCORE):
            terminated = True
            self.game_state = 'GAMEOVER'
            if self.score >= self.WIN_SCORE:
                reward += 100
            else:
                reward -= 10
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_state = 'GAMEOVER'

        truncated = False # Gymnasium standard
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # Movement
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)

        # Hint
        if shift_held and not self.shift_was_held and self.hint_timer <= 0:
            self._find_best_match()
            self.hint_timer = 60 # Show hint for 2 seconds
        self.shift_was_held = shift_held

        # Match Action
        if space_held and not self.space_was_held:
            cx, cy = self.cursor_pos
            connected_fruits = self._find_connected(cy, cx)
            
            self.moves_left -= 1
            if len(connected_fruits) >= 3:
                # Successful match
                match_score = len(connected_fruits)
                combo_bonus = 5 * self.combo_multiplier
                reward += match_score + combo_bonus
                self.score += match_score * 10 * self.combo_multiplier
                self.last_match_reward = match_score + combo_bonus
                
                self._start_match(connected_fruits)
            else:
                # Invalid move
                reward -= 1
                self.combo_multiplier = 1 # Reset combo on failed move

        self.space_was_held = space_held
        return reward

    def _populate_grid(self):
        self.grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=int)
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                possible_fruits = list(range(1, self.NUM_FRUIT_TYPES + 1))
                while True:
                    self.grid[r, c] = self.np_random.choice(possible_fruits)
                    if not self._check_match_at(r, c):
                        break
                    # If it creates a match, try another fruit type
                    possible_fruits.remove(self.grid[r, c])
                    if not possible_fruits: # Should be very rare
                        self.grid[r, c] = self.np_random.integers(1, self.NUM_FRUIT_TYPES + 1)
                        break

    def _create_visual_grid(self):
        self.visual_grid = [[{} for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                self.visual_grid[r][c] = {
                    'type': self.grid[r, c],
                    'y_offset': 0,
                    'scale': 1.0,
                    'alpha': 255
                }

    def _check_match_at(self, r, c):
        if self.grid[r, c] == 0: return False
        # Simplified check for initial population, full check is _find_connected
        fruit_type = self.grid[r, c]
        # Check horizontal
        if c > 1 and self.grid[r, c-1] == fruit_type and self.grid[r, c-2] == fruit_type:
            return True
        # Check vertical
        if r > 1 and self.grid[r-1, c] == fruit_type and self.grid[r-2, c] == fruit_type:
            return True
        return False
        
    def _find_connected(self, r, c):
        if r < 0 or r >= self.GRID_ROWS or c < 0 or c >= self.GRID_COLS or self.grid[r, c] == 0:
            return set()
        
        fruit_type = self.grid[r, c]
        q = deque([(r, c)])
        visited = set([(r, c)])
        
        while q:
            curr_r, curr_c = q.popleft()
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS and (nr, nc) not in visited:
                    if self.grid[nr, nc] == fruit_type:
                        visited.add((nr, nc))
                        q.append((nr, nc))
        return visited

    def _start_match(self, cells):
        self.game_state = 'MATCH'
        self.animation_timer = 15 # frames
        self.matched_cells_for_anim = cells
        
        for r, c in cells:
            self.grid[r, c] = 0
            for _ in range(10):
                self.particles.append(self._create_particle(c, r))
        
        # Add floating text for score
        center_c = sum(c for r,c in cells) / len(cells)
        center_r = sum(r for r,c in cells) / len(cells)
        text = f"+{int(self.last_match_reward)}"
        if self.combo_multiplier > 1:
            text = f"x{self.combo_multiplier} COMBO!"
        self.floating_texts.append({
            'text': text,
            'pos': [self.GRID_X + center_c * self.CELL_SIZE, self.GRID_Y + center_r * self.CELL_SIZE],
            'life': 45,
            'color': (255, 255, 100)
        })

    def _start_fall(self):
        self.game_state = 'FALL'
        self.animation_timer = 15 # frames
        self.fall_map = {}
        
        for c in range(self.GRID_COLS):
            empty_count = 0
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r, c] == 0:
                    empty_count += 1
                elif empty_count > 0:
                    # This fruit needs to fall
                    self.fall_map[(r, c)] = empty_count
                    self.visual_grid[r+empty_count][c] = self.visual_grid[r][c]
                    self.visual_grid[r+empty_count][c]['y_offset'] = -empty_count * self.CELL_SIZE

    def _apply_fall(self):
        new_grid = np.zeros_like(self.grid)
        for c in range(self.GRID_COLS):
            write_r = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r, c] != 0:
                    new_grid[write_r, c] = self.grid[r, c]
                    write_r -= 1
        self.grid = new_grid

    def _start_refill(self):
        self.game_state = 'REFILL'
        self.animation_timer = 15 # frames
        
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r, c] == 0:
                    self.grid[r, c] = self.np_random.integers(1, self.NUM_FRUIT_TYPES + 1)
                    self.visual_grid[r][c] = {
                        'type': self.grid[r, c],
                        'y_offset': -self.CELL_SIZE,
                        'scale': 1.0,
                        'alpha': 255
                    }

    def _check_for_chains(self):
        all_matches = set()
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if (r, c) not in all_matches:
                    connected = self._find_connected(r, c)
                    if len(connected) >= 3:
                        all_matches.update(connected)
        
        if all_matches:
            self.combo_multiplier += 1
            match_score = len(all_matches)
            combo_bonus = 5 * self.combo_multiplier
            reward = match_score + combo_bonus
            self.score += match_score * 10 * self.combo_multiplier
            self.last_match_reward = reward
            
            self._start_match(all_matches)
            return reward
        else:
            self.game_state = 'IDLE'
            self.combo_multiplier = 1
            return 0
    
    def _find_best_match(self):
        best_match = set()
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r,c] != 0:
                    match = self._find_connected(r, c)
                    if len(match) > len(best_match):
                        best_match = match
        self.hint_cells = best_match

    def _update_animations(self):
        if self.animation_timer > 0:
            self.animation_timer -= 1
        
        if self.hint_timer > 0:
            self.hint_timer -= 1
            if self.hint_timer == 0:
                self.hint_cells = set()

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['vel'][1] += 0.1 # Gravity

        # Update floating texts
        self.floating_texts = [t for t in self.floating_texts if t['life'] > 0]
        for t in self.floating_texts:
            t['pos'][1] -= 0.5
            t['life'] -= 1

        # Animate visual grid
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                v_fruit = self.visual_grid[r][c]
                # Fall/Refill animation
                v_fruit['y_offset'] *= 0.8
                if abs(v_fruit['y_offset']) < 0.5: v_fruit['y_offset'] = 0
                
                # Match animation
                if self.game_state == 'MATCH' and (r,c) in self.matched_cells_for_anim:
                    v_fruit['scale'] *= 0.85
                    v_fruit['alpha'] *= 0.85
                elif v_fruit['scale'] < 1.0:
                    v_fruit['scale'] = 1.0
                    v_fruit['alpha'] = 255


    def _create_particle(self, c, r):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(2, 5)
        fruit_type = self.visual_grid[r][c]['type']
        return {
            'pos': [self.GRID_X + (c + 0.5) * self.CELL_SIZE, self.GRID_Y + (r + 0.5) * self.CELL_SIZE],
            'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
            'life': self.np_random.integers(20, 40),
            'color': self.PARTICLE_COLORS[fruit_type - 1],
            'size': self.np_random.uniform(3, 7)
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_surface = pygame.Surface((self.GRID_WIDTH, self.GRID_HEIGHT), pygame.SRCALPHA)
        grid_surface.fill((0,0,0,0))
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                rect = pygame.Rect(c * self.CELL_SIZE, r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(grid_surface, self.COLOR_GRID, rect, 1)
        self.screen.blit(grid_surface, (self.GRID_X, self.GRID_Y))

        # Highlight for current selection
        if self.game_state == 'IDLE':
            cx, cy = self.cursor_pos
            current_selection = self._find_connected(cy, cx)
            if len(current_selection) >= 3:
                for r, c in current_selection:
                    highlight_rect = pygame.Rect(self.GRID_X + c * self.CELL_SIZE, self.GRID_Y + r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                    s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                    alpha = 100 + 50 * math.sin(self.steps * 0.2)
                    s.fill((255, 255, 255, alpha))
                    self.screen.blit(s, highlight_rect.topleft)

        # Draw fruits
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                v_fruit = self.visual_grid[r][c]
                fruit_type = v_fruit['type']
                if fruit_type == 0: continue
                
                radius = int((self.CELL_SIZE * 0.8 / 2) * v_fruit['scale'])
                if radius <= 0: continue

                center_x = self.GRID_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
                center_y = self.GRID_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2 + v_fruit['y_offset']
                
                # Bobbing animation
                if self.game_state == 'IDLE':
                    center_y += 2 * math.sin((self.steps + c*2 + r*2) * 0.1)

                color = self.FRUIT_COLORS[fruit_type - 1]
                
                # Cast coordinates to int for gfxdraw
                int_center_x = int(center_x)
                int_center_y = int(center_y)
                
                # Use gfxdraw for antialiasing
                pygame.gfxdraw.filled_circle(self.screen, int_center_x, int_center_y, radius, (*color, int(v_fruit['alpha'])))
                pygame.gfxdraw.aacircle(self.screen, int_center_x, int_center_y, radius, (*color, int(v_fruit['alpha'])))
                
                # Highlight sheen
                sheen_radius = int(radius * 0.3)
                sheen_x = int(center_x - radius * 0.3)
                sheen_y = int(center_y - radius * 0.3)
                pygame.gfxdraw.filled_circle(self.screen, sheen_x, sheen_y, sheen_radius, (255, 255, 255, 50))


        # Draw hint
        if self.hint_timer > 0:
            for r, c in self.hint_cells:
                rect = pygame.Rect(self.GRID_X + c * self.CELL_SIZE, self.GRID_Y + r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, (255, 255, 100), rect, 3, border_radius=5)

        # Draw cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(self.GRID_X + cx * self.CELL_SIZE, self.GRID_Y + cy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pulse = 1 + 0.1 * math.sin(self.steps * 0.3)
        pygame.draw.rect(self.screen, (255, 255, 255), cursor_rect.inflate(pulse*4, pulse*4), 2, border_radius=5)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, p['life'] * 10))
            color = (*p['color'], alpha)
            size = int(p['size'] * (p['life'] / 40.0))
            if size > 0:
                pygame.draw.rect(self.screen, color, (p['pos'][0], p['pos'][1], size, size))

        # Draw floating texts
        for t in self.floating_texts:
            alpha = max(0, min(255, t['life'] * 5))
            text_surf = self.font_combo.render(t['text'], True, (*t['color'][:3], alpha))
            text_rect = text_surf.get_rect(center=t['pos'])
            self.screen.blit(text_surf, text_rect)


    def _render_ui(self):
        # UI Background
        ui_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, 50)
        s = pygame.Surface((self.SCREEN_WIDTH, 50), pygame.SRCALPHA)
        s.fill(self.COLOR_UI_BG)
        self.screen.blit(s, (0,0))
        pygame.draw.line(self.screen, (80, 75, 100), (0, 50), (self.SCREEN_WIDTH, 50), 2)

        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 15))

        # Moves
        moves_text = self.font_ui.render(f"MOVES: {self.moves_left}", True, self.COLOR_UI_TEXT)
        moves_rect = moves_text.get_rect(centerx=self.SCREEN_WIDTH / 2, y=15)
        self.screen.blit(moves_text, moves_rect)
        
        # Goal
        goal_text = self.font_ui.render(f"GOAL: {self.WIN_SCORE}", True, self.COLOR_UI_TEXT)
        goal_rect = goal_text.get_rect(right=self.SCREEN_WIDTH - 20, y=15)
        self.screen.blit(goal_text, goal_rect)

        # Game Over Screen
        if self.game_state == 'GAMEOVER':
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            color = (100, 255, 100) if self.score >= self.WIN_SCORE else (255, 100, 100)
            
            end_text = self.font_game_over.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "combo": self.combo_multiplier,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not run in the headless verification environment
    env = GameEnv()
    obs, info = env.reset()
    
    # Un-set the headless environment variable for local display
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Fruit Matcher")
    
    running = True
    total_reward = 0
    
    # Game loop
    while running:
        # Action defaults to no-op
        action = [0, 0, 0] 
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get key presses for human control
        keys = pygame.key.get_pressed()
        
        # Movement (only one direction at a time)
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        # Space and Shift
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            total_reward = 0

        # Control frame rate
        env.clock.tick(30)

    env.close()