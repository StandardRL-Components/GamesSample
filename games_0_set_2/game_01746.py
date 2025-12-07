
# Generated: 2025-08-27T18:10:23.564298
# Source Brief: brief_01746.md
# Brief Index: 1746

        
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

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to select a gem, "
        "then move to an adjacent gem and press space again to swap."
    )

    game_description = (
        "A classic match-3 puzzle game. Swap adjacent gems to create lines of 3 or more. "
        "Chain reactions create combos for bonus points. Reach the target score before you run out of moves!"
    )

    auto_advance = True

    # --- Constants ---
    GRID_WIDTH, GRID_HEIGHT = 8, 8
    GEM_TYPES = 6  # 1 to 6
    GEM_SIZE = 40
    GRID_OFFSET_X = (640 - GRID_WIDTH * GEM_SIZE) // 2
    GRID_OFFSET_Y = (400 - GRID_HEIGHT * GEM_SIZE) // 2 + 20

    TARGET_SCORE = 1000
    MAX_MOVES = 20
    
    REWARD_GEM_MATCH = 1
    REWARD_COMBO = 10
    REWARD_INVALID_SWAP = -0.1
    REWARD_WIN = 100
    REWARD_LOSE = -1

    ANIMATION_SPEED = 0.15 # Progress per frame

    # --- Colors ---
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 80)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_CURSOR_SELECTED = (255, 255, 0)
    COLOR_TEXT = (220, 220, 240)
    GEM_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (255, 80, 255),  # Magenta
        (80, 255, 255),  # Cyan
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Game state variables
        self.grid = None
        self.cursor_pos = None
        self.selected_pos = None
        self.score = 0
        self.moves_remaining = 0
        self.game_over = False
        self.steps = 0
        
        # Animation and input handling
        self.game_phase = "IDLE" # IDLE, SWAPPING, MATCHING, FALLING, REVERSING
        self.animations = []
        self.particles = []
        self.prev_space_press = False
        self.turn_reward = 0
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._create_board()
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_pos = None
        
        self.score = 0
        self.moves_remaining = self.MAX_MOVES
        self.game_over = False
        self.steps = 0
        
        self.game_phase = "IDLE"
        self.animations.clear()
        self.particles.clear()
        self.prev_space_press = False
        self.turn_reward = 0
        
        return self._get_observation(), self._get_info()

    def _create_board(self):
        self.grid = self.np_random.integers(1, self.GEM_TYPES + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        while self._find_all_matches() or not self._find_possible_moves():
            self.grid = self.np_random.integers(1, self.GEM_TYPES + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = self._update_animations_and_game_phase(action)
        
        terminated = self.game_over
        if terminated:
            if self.score >= self.TARGET_SCORE:
                reward += self.REWARD_WIN
            else:
                reward += self.REWARD_LOSE

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_animations_and_game_phase(self, action):
        movement, space_press, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # --- Update existing animations ---
        if self.animations:
            for anim in self.animations:
                anim['progress'] += self.ANIMATION_SPEED
            self.animations = [anim for anim in self.animations if anim['progress'] < 1.0]

        # --- Update particles ---
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

        # --- Phase-based logic ---
        if self.game_phase == "IDLE":
            if not self.animations: # Ensure no lingering animations
                reward = self._handle_player_input(movement, space_press)
        
        elif not self.animations: # Transition between animation phases
            if self.game_phase == "SWAPPING":
                self._resolve_swap()
            elif self.game_phase == "REVERSING":
                self._resolve_reverse_swap()
            elif self.game_phase == "MATCHING":
                self._resolve_matches()
            elif self.game_phase == "FALLING":
                reward = self._resolve_fall()

        self.prev_space_press = space_press
        return reward

    def _handle_player_input(self, movement, space_press):
        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)

        # --- Selection/Swap ---
        is_space_click = space_press and not self.prev_space_press
        if is_space_click:
            # SFX: select_gem.wav
            if self.selected_pos is None:
                self.selected_pos = list(self.cursor_pos)
            else:
                x1, y1 = self.selected_pos
                x2, y2 = self.cursor_pos
                if (x1, y1) == (x2, y2): # Deselect
                    self.selected_pos = None
                elif abs(x1 - x2) + abs(y1 - y2) == 1: # Is adjacent
                    self.moves_remaining -= 1
                    self.animations.append({
                        "type": "SWAP", "pos1": (x1, y1), "pos2": (x2, y2), "progress": 0.0
                    })
                    self.game_phase = "SWAPPING"
                    self.turn_reward = 0 # Reset reward for the new turn
                    # SFX: swap.wav
                else: # Invalid non-adjacent selection
                    self.selected_pos = list(self.cursor_pos) # Select new gem
        return 0
    
    def _resolve_swap(self):
        # This function is called when the swap animation finishes
        x1, y1 = self.selected_pos
        x2, y2 = self.cursor_pos
        self.grid[y1, x1], self.grid[y2, x2] = self.grid[y2, x2], self.grid[y1, x1]

        matches = self._find_all_matches()
        if not matches: # Invalid swap
            self.turn_reward += self.REWARD_INVALID_SWAP
            self.animations.append({
                "type": "SWAP", "pos1": (x1, y1), "pos2": (x2, y2), "progress": 0.0
            })
            self.game_phase = "REVERSING"
            # SFX: invalid_swap.wav
        else:
            self._start_match_sequence(matches)
        
    def _resolve_reverse_swap(self):
        # Swap back logically
        x1, y1 = self.selected_pos
        x2, y2 = self.cursor_pos
        self.grid[y1, x1], self.grid[y2, x2] = self.grid[y2, x2], self.grid[y1, x1]
        self.selected_pos = None
        self.game_phase = "IDLE"
        if self.moves_remaining <= 0:
            self.game_over = True

    def _start_match_sequence(self, matches, is_combo=False):
        self.game_phase = "MATCHING"
        if is_combo:
            self.turn_reward += self.REWARD_COMBO
            # SFX: combo.wav
        
        for r, c in matches:
            self.turn_reward += self.REWARD_GEM_MATCH
            self.score += 10 # Simple score increment
            self.animations.append({
                "type": "DESTROY", "pos": (c, r), "gem_type": self.grid[r, c], "progress": 0.0
            })
            # SFX: match.wav
            self._spawn_particles(c, r, self.grid[r, c])
        
        self.grid[tuple(zip(*matches))] = 0 # Set matched gems to 0
        self.selected_pos = None

    def _spawn_particles(self, c, r, gem_type):
        px, py = self.GRID_OFFSET_X + c * self.GEM_SIZE + self.GEM_SIZE / 2, self.GRID_OFFSET_Y + r * self.GEM_SIZE + self.GEM_SIZE / 2
        color = self.GEM_COLORS[gem_type - 1]
        for _ in range(10):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'x': px, 'y': py, 'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'life': random.randint(15, 30), 'color': color
            })

    def _resolve_matches(self):
        self.game_phase = "FALLING"
        fall_anims, self.grid = self._apply_gravity_and_refill()
        self.animations.extend(fall_anims)
        # SFX: gems_fall.wav

    def _resolve_fall(self):
        matches = self._find_all_matches()
        if matches:
            self._start_match_sequence(matches, is_combo=True)
            return 0 # Reward is handled in the match sequence
        else:
            final_turn_reward = self.turn_reward
            self.turn_reward = 0
            self.game_phase = "IDLE"
            if not self._find_possible_moves():
                # SFX: reshuffle.wav
                self._create_board() # Reshuffle if no moves
            if self.moves_remaining <= 0 or self.score >= self.TARGET_SCORE:
                self.game_over = True
            return final_turn_reward

    def _apply_gravity_and_refill(self):
        fall_anims = []
        new_grid = np.zeros_like(self.grid)
        for c in range(self.GRID_WIDTH):
            write_r = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if r != write_r:
                        fall_anims.append({
                            "type": "FALL", "start_row": r, "end_row": write_r, "col": c,
                            "gem_type": self.grid[r, c], "progress": 0.0
                        })
                    new_grid[write_r, c] = self.grid[r, c]
                    write_r -= 1
            # Refill
            for r in range(write_r, -1, -1):
                gem = self.np_random.integers(1, self.GEM_TYPES + 1)
                new_grid[r, c] = gem
                fall_anims.append({
                    "type": "FALL", "start_row": - (write_r - r + 1), "end_row": r, "col": c,
                    "gem_type": gem, "progress": 0.0
                })
        return fall_anims, new_grid

    def _find_all_matches(self):
        matches = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH - 2):
                if self.grid[r, c] != 0 and self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2]:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT - 2):
                if self.grid[r, c] != 0 and self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c]:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return matches

    def _find_possible_moves(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                # Check swap right
                if c < self.GRID_WIDTH - 1:
                    self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                    if self._find_all_matches():
                        self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                        return True
                    self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                # Check swap down
                if r < self.GRID_HEIGHT - 1:
                    self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
                    if self._find_all_matches():
                        self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
                        return True
                    self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_OFFSET_X, self.GRID_OFFSET_Y, self.GRID_WIDTH * self.GEM_SIZE, self.GRID_HEIGHT * self.GEM_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)
        
        # Keep track of gems handled by animations
        animated_gems = set()
        for anim in self.animations:
            if anim['type'] == 'SWAP':
                animated_gems.add(anim['pos1'])
                animated_gems.add(anim['pos2'])
            elif anim['type'] == 'DESTROY':
                animated_gems.add(anim['pos'])
            elif anim['type'] == 'FALL':
                # Only add the destination, as the source is empty
                animated_gems.add((anim['col'], anim['end_row']))

        # Draw static gems
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if (c, r) not in animated_gems and self.grid[r, c] != 0:
                    self._draw_gem(c, r, self.grid[r, c])

        # Draw animated gems
        for anim in self.animations:
            p = anim['progress']
            if anim['type'] == 'SWAP':
                x1, y1 = anim['pos1']; x2, y2 = anim['pos2']
                gem1_type = self.grid[y2, x2] # After swap
                gem2_type = self.grid[y1, x1] # After swap
                # Interpolate positions
                ix1 = x1 + (x2 - x1) * p; iy1 = y1 + (y2 - y1) * p
                ix2 = x2 + (x1 - x2) * p; iy2 = y2 + (y1 - y2) * p
                self._draw_gem(ix1, iy1, gem1_type)
                self._draw_gem(ix2, iy2, gem2_type)
            elif anim['type'] == 'DESTROY':
                scale = 1.0 - p
                alpha = 255 * (1.0 - p)
                self._draw_gem(anim['pos'][0], anim['pos'][1], anim['gem_type'], scale, alpha)
            elif anim['type'] == 'FALL':
                y = anim['start_row'] + (anim['end_row'] - anim['start_row']) * p
                self._draw_gem(anim['col'], y, anim['gem_type'])

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, p['life'] * 15))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), 2, color)

        # Draw cursor
        cursor_color = self.COLOR_CURSOR_SELECTED if self.selected_pos else self.COLOR_CURSOR
        cursor_rect = pygame.Rect(
            self.GRID_OFFSET_X + self.cursor_pos[0] * self.GEM_SIZE,
            self.GRID_OFFSET_Y + self.cursor_pos[1] * self.GEM_SIZE,
            self.GEM_SIZE, self.GEM_SIZE
        )
        pygame.draw.rect(self.screen, cursor_color, cursor_rect, 3)
        if self.selected_pos:
            selected_rect = pygame.Rect(
                self.GRID_OFFSET_X + self.selected_pos[0] * self.GEM_SIZE,
                self.GRID_OFFSET_Y + self.selected_pos[1] * self.GEM_SIZE,
                self.GEM_SIZE, self.GEM_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_CURSOR_SELECTED, selected_rect, 3)

    def _draw_gem(self, c, r, gem_type, scale=1.0, alpha=255.0):
        if gem_type == 0: return
        size = int(self.GEM_SIZE * scale)
        if size <= 0: return

        center_x = self.GRID_OFFSET_X + c * self.GEM_SIZE + self.GEM_SIZE / 2
        center_y = self.GRID_OFFSET_Y + r * self.GEM_SIZE + self.GEM_SIZE / 2
        
        gem_rect = pygame.Rect(center_x - size/2, center_y - size/2, size, size)
        
        color = self.GEM_COLORS[gem_type - 1]
        
        # Create a surface for transparency
        temp_surface = pygame.Surface((size, size), pygame.SRCALPHA)
        
        # Draw gem shape (octagon)
        points = [
            (size * 0.3, 0), (size * 0.7, 0), (size, size * 0.3), (size, size * 0.7),
            (size * 0.7, size), (size * 0.3, size), (0, size * 0.7), (0, size * 0.3)
        ]
        pygame.draw.polygon(temp_surface, (*color, alpha), points)
        
        # Draw highlight
        highlight_color = (min(255, color[0]+60), min(255, color[1]+60), min(255, color[2]+60))
        pygame.draw.polygon(temp_surface, (*highlight_color, alpha), [points[0], points[1], points[7]])

        self.screen.blit(temp_surface, gem_rect.topleft)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        # Moves
        moves_text = self.font_large.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (640 - moves_text.get_width() - 10, 10))
        # Game Over
        if self.game_over:
            overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_lose_text = "You Win!" if self.score >= self.TARGET_SCORE else "Game Over"
            text_surf = self.font_large.render(win_lose_text, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(320, 180))
            self.screen.blit(text_surf, text_rect)
            
            final_score_surf = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            final_score_rect = final_score_surf.get_rect(center=(320, 220))
            self.screen.blit(final_score_surf, final_score_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_remaining": self.moves_remaining}

    def close(self):
        pygame.quit()
    
    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Gem Swap")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0
        space = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        
        action = [movement, space, 0] # Shift is not used
        obs, reward, terminated, truncated, info = env.step(action)

        if reward != 0:
            print(f"Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_remaining']}")

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print("Game Over!")
            pygame.time.wait(3000)
            obs, info = env.reset()

        clock.tick(30) # Match the auto_advance rate

    env.close()