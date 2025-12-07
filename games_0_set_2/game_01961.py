
# Generated: 2025-08-28T03:18:01.652631
# Source Brief: brief_01961.md
# Brief Index: 1961

        
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


# A simple particle class for visual effects
class Particle:
    def __init__(self, x, y, color, np_random):
        self.x = x
        self.y = y
        self.color = color
        self.vx = np_random.uniform(-2, 2)
        self.vy = np_random.uniform(-4, -1)
        self.life = np_random.integers(20, 40)
        self.radius = np_random.uniform(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.2  # Gravity
        self.life -= 1
        self.radius -= 0.1

    def draw(self, surface):
        if self.life > 0 and self.radius > 0:
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), int(self.radius))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the selector. Press space to select a tile, "
        "then move to an adjacent tile and press space again to swap. Press shift to cancel a selection."
    )

    game_description = (
        "Swap adjacent colored tiles to create matches of 3 or more. "
        "Clear the whole board to win. Chain reactions grant score multipliers!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 8
        self.TILE_SIZE = 40
        self.GRID_MARGIN_X = (self.WIDTH - self.GRID_SIZE * self.TILE_SIZE) // 2
        self.GRID_MARGIN_Y = (self.HEIGHT - self.GRID_SIZE * self.TILE_SIZE) // 2
        self.MAX_STEPS = 500
        self.MOVE_COOLDOWN_FRAMES = 4
        
        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_TEXT = (220, 230, 255)
        self.TILE_COLORS = [
            (0, 0, 0), # 0 is empty
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
            (255, 150, 50),  # Orange
        ]
        self.COLOR_SELECTOR = (255, 255, 255)
        self.COLOR_SELECTED = (255, 255, 0)
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        
        # --- State Variables ---
        self.grid = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.selector_pos = [0, 0]
        self.selected_tile_pos = None
        self.logic_is_processing = False
        self.animations = []
        self.particles = []
        self.chain_multiplier = 1
        self.last_action_state = {'space': 0, 'shift': 0}
        self.move_cooldown = 0
        
        # Initialize state
        self.reset()
        
        # --- Validation ---
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.selector_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_tile_pos = None
        self.logic_is_processing = False
        self.animations = []
        self.particles = []
        self.chain_multiplier = 1
        self.last_action_state = {'space': 0, 'shift': 0}
        self.move_cooldown = 0
        
        self._generate_board_with_guaranteed_moves()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        if not self.game_over:
            if not self.logic_is_processing:
                reward += self._handle_input(action)
            
            reward += self._update_game_logic()

        self.steps += 1
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            self.game_over = True
            # Check for win condition
            if np.sum(self.grid > 0) == 0:
                reward += 100
            else: # Loss condition (max steps or no moves after shuffle)
                reward -= 100
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_val, shift_val = action[0], action[1], action[2]
        space_pressed = space_val == 1 and self.last_action_state['space'] == 0
        shift_pressed = shift_val == 1 and self.last_action_state['shift'] == 0
        self.last_action_state = {'space': space_val, 'shift': shift_val}

        # --- Selector Movement ---
        if self.move_cooldown > 0:
            self.move_cooldown -= 1
        
        if movement != 0 and self.move_cooldown == 0:
            self.move_cooldown = self.MOVE_COOLDOWN_FRAMES
            if movement == 1: self.selector_pos[0] = max(0, self.selector_pos[0] - 1)
            elif movement == 2: self.selector_pos[0] = min(self.GRID_SIZE - 1, self.selector_pos[0] + 1)
            elif movement == 3: self.selector_pos[1] = max(0, self.selector_pos[1] - 1)
            elif movement == 4: self.selector_pos[1] = min(self.GRID_SIZE - 1, self.selector_pos[1] + 1)

        # --- Shift to Cancel ---
        if shift_pressed and self.selected_tile_pos:
            self.selected_tile_pos = None
            # sfx: cancel_select
            return 0

        # --- Space to Select/Swap ---
        if space_pressed:
            r, c = self.selector_pos
            if self.grid[r, c] == 0: # Cannot select empty space
                return 0
            
            if not self.selected_tile_pos:
                self.selected_tile_pos = list(self.selector_pos)
                # sfx: select_tile
            else:
                # Check for adjacency
                r1, c1 = self.selected_tile_pos
                r2, c2 = self.selector_pos
                if abs(r1 - r2) + abs(c1 - c2) == 1:
                    return self._attempt_swap(r1, c1, r2, c2)
                else: # Not adjacent, make current selection the new one
                    self.selected_tile_pos = list(self.selector_pos)
                    # sfx: select_tile
        return 0

    def _attempt_swap(self, r1, c1, r2, c2):
        # Perform swap
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
        
        matches1 = self._find_matches_at(r1, c1)
        matches2 = self._find_matches_at(r2, c2)
        all_matches = matches1.union(matches2)

        if all_matches:
            # sfx: valid_swap
            self.logic_is_processing = True
            self.chain_multiplier = 1
            self.selected_tile_pos = None
            
            reward = self._clear_matches(all_matches)
            self._apply_gravity_and_refill()
            return reward
        else:
            # Invalid swap, swap back
            self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
            self.selected_tile_pos = None
            # sfx: invalid_swap
            return -1

    def _update_game_logic(self):
        reward = 0
        # Update animations
        self.animations = [anim for anim in self.animations if not anim['done']]
        for anim in self.animations:
            anim['progress'] += 0.15 # Animation speed
            if anim['progress'] >= 1.0:
                anim['progress'] = 1.0
                anim['done'] = True
        
        # Update particles
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.life > 0]

        # If all animations are done, check for chain reactions
        if self.logic_is_processing and not self.animations:
            all_matches = self._find_all_matches()
            if all_matches:
                self.chain_multiplier += 1
                # sfx: chain_reaction
                reward += self._clear_matches(all_matches) * self.chain_multiplier
                self._apply_gravity_and_refill()
            else:
                # Chain is over
                self.logic_is_processing = False
                self.chain_multiplier = 1
                # Check for possible moves and shuffle if necessary
                if not self._find_possible_moves() and np.sum(self.grid > 0) > 0:
                    self._shuffle_board()
                    # sfx: shuffle
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_grid()
        self._draw_tiles()
        self._draw_selector()
        for p in self.particles:
            p.draw(self.screen)

    def _draw_grid(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                rect = pygame.Rect(self.GRID_MARGIN_X + c * self.TILE_SIZE,
                                   self.GRID_MARGIN_Y + r * self.TILE_SIZE,
                                   self.TILE_SIZE, self.TILE_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

    def _draw_tiles(self):
        animating_tiles = set()
        for anim in self.animations:
            if anim['type'] == 'fall':
                r_start, c_start = anim['from_pos']
                r_end, c_end = anim['to_pos']
                
                interp_r = r_start + (r_end - r_start) * anim['progress']
                interp_c = c_start + (c_end - c_start) * anim['progress']
                
                self._draw_tile(interp_c, interp_r, anim['color_idx'])
                animating_tiles.add((r_end, c_end))

        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] > 0 and (r, c) not in animating_tiles:
                    self._draw_tile(c, r, self.grid[r, c])

    def _draw_tile(self, c, r, color_idx, scale=1.0):
        center_x = self.GRID_MARGIN_X + c * self.TILE_SIZE + self.TILE_SIZE / 2
        center_y = self.GRID_MARGIN_Y + r * self.TILE_SIZE + self.TILE_SIZE / 2
        radius = int(self.TILE_SIZE / 2.2 * scale)
        color = self.TILE_COLORS[color_idx]
        
        # Draw a slightly darker background for depth
        darker_color = tuple(max(0, val - 40) for val in color)
        pygame.gfxdraw.filled_circle(self.screen, int(center_x), int(center_y), radius, darker_color)
        pygame.gfxdraw.aacircle(self.screen, int(center_x), int(center_y), radius, darker_color)
        
        # Draw the main tile body
        inner_radius = int(radius * 0.8)
        pygame.gfxdraw.filled_circle(self.screen, int(center_x), int(center_y), inner_radius, color)
        pygame.gfxdraw.aacircle(self.screen, int(center_x), int(center_y), inner_radius, color)

    def _draw_selector(self):
        # Draw main selector
        r, c = self.selector_pos
        rect = pygame.Rect(self.GRID_MARGIN_X + c * self.TILE_SIZE,
                           self.GRID_MARGIN_Y + r * self.TILE_SIZE,
                           self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, rect, 3, border_radius=5)

        # Draw selected tile highlight
        if self.selected_tile_pos:
            r_sel, c_sel = self.selected_tile_pos
            rect_sel = pygame.Rect(self.GRID_MARGIN_X + c_sel * self.TILE_SIZE,
                                   self.GRID_MARGIN_Y + r_sel * self.TILE_SIZE,
                                   self.TILE_SIZE, self.TILE_SIZE)
            
            # Pulsating effect
            pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2
            color = (
                int(255 * (1 - pulse) + self.COLOR_SELECTED[0] * pulse),
                int(255 * (1 - pulse) + self.COLOR_SELECTED[1] * pulse),
                int(0 * (1 - pulse) + self.COLOR_SELECTED[2] * pulse),
            )
            pygame.draw.rect(self.screen, color, rect_sel, 4, border_radius=8)

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        moves_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 10, 10))

        if self.game_over:
            win = np.sum(self.grid > 0) == 0
            msg = "BOARD CLEARED!" if win else "GAME OVER"
            color = (100, 255, 100) if win else (255, 100, 100)
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _generate_board_with_guaranteed_moves(self):
        while True:
            self.grid = self.np_random.integers(1, len(self.TILE_COLORS), size=(self.GRID_SIZE, self.GRID_SIZE))
            if self._find_possible_moves():
                break

    def _find_matches_at(self, r, c):
        if r < 0 or r >= self.GRID_SIZE or c < 0 or c >= self.GRID_SIZE or self.grid[r, c] == 0:
            return set()

        color = self.grid[r, c]
        
        # Horizontal match
        h_match = {(r, c)}
        # Left
        for i in range(c - 1, -1, -1):
            if self.grid[r, i] == color: h_match.add((r, i))
            else: break
        # Right
        for i in range(c + 1, self.GRID_SIZE):
            if self.grid[r, i] == color: h_match.add((r, i))
            else: break

        # Vertical match
        v_match = {(r, c)}
        # Up
        for i in range(r - 1, -1, -1):
            if self.grid[i, c] == color: v_match.add((i, c))
            else: break
        # Down
        for i in range(r + 1, self.GRID_SIZE):
            if self.grid[i, c] == color: v_match.add((i, c))
            else: break
        
        matches = set()
        if len(h_match) >= 3: matches.update(h_match)
        if len(v_match) >= 3: matches.update(v_match)
        
        return matches

    def _find_all_matches(self):
        all_matches = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                all_matches.update(self._find_matches_at(r, c))
        return all_matches

    def _clear_matches(self, matches):
        reward = 0
        if not matches:
            return 0
        
        num_cleared = len(matches)
        reward += num_cleared
        if num_cleared > 3:
            reward += 5 # Bonus for clearing more than 3
        
        self.score += reward * self.chain_multiplier

        for r, c in matches:
            # Create particles
            for _ in range(10):
                px = self.GRID_MARGIN_X + c * self.TILE_SIZE + self.TILE_SIZE / 2
                py = self.GRID_MARGIN_Y + r * self.TILE_SIZE + self.TILE_SIZE / 2
                self.particles.append(Particle(px, py, self.TILE_COLORS[self.grid[r, c]], self.np_random))
            self.grid[r, c] = 0
        
        # sfx: match_clear
        return reward

    def _apply_gravity_and_refill(self):
        for c in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if r != empty_row:
                        self.animations.append({
                            'type': 'fall', 'from_pos': [r, c], 'to_pos': [empty_row, c],
                            'color_idx': self.grid[r, c], 'progress': 0.0, 'done': False
                        })
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                    empty_row -= 1
            
            # Refill top with new tiles
            for r in range(empty_row, -1, -1):
                new_color = self.np_random.integers(1, len(self.TILE_COLORS))
                self.grid[r, c] = new_color
                self.animations.append({
                    'type': 'fall', 'from_pos': [r - (empty_row + 1), c], 'to_pos': [r, c],
                    'color_idx': new_color, 'progress': 0.0, 'done': False
                })

    def _find_possible_moves(self):
        moves = []
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] == 0: continue
                # Check swap right
                if c < self.GRID_SIZE - 1 and self.grid[r, c+1] != 0:
                    self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                    if self._find_matches_at(r, c) or self._find_matches_at(r, c+1):
                        moves.append(((r, c), (r, c+1)))
                    self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                # Check swap down
                if r < self.GRID_SIZE - 1 and self.grid[r+1, c] != 0:
                    self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
                    if self._find_matches_at(r, c) or self._find_matches_at(r+1, c):
                        moves.append(((r, c), (r+1, c)))
                    self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
        return moves

    def _shuffle_board(self):
        while True:
            flat_grid = self.grid.flatten()
            self.np_random.shuffle(flat_grid)
            self.grid = flat_grid.reshape((self.GRID_SIZE, self.GRID_SIZE))
            if self._find_possible_moves():
                # Clear animations as board state is now instant
                self.animations = []
                break
    
    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            return True
        if np.sum(self.grid > 0) == 0: # All tiles cleared
            return True
        # If we are not processing logic and there are no moves, it's a loss
        # The shuffle should prevent this, but as a fallback.
        if not self.logic_is_processing and not self._find_possible_moves():
            return True
        return False

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # To play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    
    # Create a window to display the game
    pygame.display.set_caption("Match-3 Gym Environment")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()

    total_reward = 0
    
    while running:
        action = [0, 0, 0] # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(30) # Run at 30 FPS

    env.close()