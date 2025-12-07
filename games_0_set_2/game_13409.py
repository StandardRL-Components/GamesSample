import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T21:25:03.910625
# Source Brief: brief_03409.md
# Brief Index: 3409
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A turn-based 5x5 grid puzzle game where the goal is to clear all colored
    blocks by swapping adjacent blocks to form lines of 3 or more. The episode
    ends when the board is cleared (win) or after 60 moves (loss).

    The environment is designed with high-quality visuals, including smooth
    animations for block movements, satisfying particle effects for clearing
    blocks, and a clean, readable UI.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Clear the board by swapping adjacent blocks to create lines of three or more of the same color."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to swap the selected block. Use space/shift to cycle through blocks."
    )
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 5
        self.CELL_SIZE = 64
        self.GRID_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_ORIGIN = (
            (self.WIDTH - self.GRID_WIDTH) // 2,
            (self.HEIGHT - self.GRID_WIDTH) // 2
        )
        self.MAX_STEPS = 60
        self.NUM_COLORS = 5
        self.ANIMATION_SPEED = 0.25 # Lerp factor

        # Colors
        self.COLOR_BG = (26, 26, 46)
        self.COLOR_GRID = (58, 58, 94)
        self.COLOR_TEXT = (224, 224, 224)
        self.COLOR_SELECTION = (255, 255, 255)
        self.BLOCK_COLORS = [
            (231, 76, 60),   # Red
            (46, 204, 113),  # Green
            (52, 152, 219),  # Blue
            (241, 196, 15),  # Yellow
            (155, 89, 182)   # Purple
        ]
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 72)
        
        # State variables are initialized in reset()
        self.blocks = []
        self.particles = []
        self.selected_pos = [0, 0]
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.previous_space_held = False
        self.previous_shift_held = False
        
        # self.reset() # reset() is called by the wrapper
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.previous_space_held = False
        self.previous_shift_held = False
        self.particles.clear()
        
        self._generate_initial_blocks()
        
        # Find first block to select
        if self.blocks:
            self.selected_pos = [self.blocks[0]['r'], self.blocks[0]['c']]
        else:
            self.selected_pos = [-1, -1]

        return self._get_observation(), self._get_info()

    def _generate_initial_blocks(self):
        self.blocks.clear()
        grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                possible_colors = list(range(1, self.NUM_COLORS + 1))
                while possible_colors:
                    color = self.np_random.choice(possible_colors)
                    # Avoid creating matches on generation
                    if r >= 2 and grid[r-1, c] == color and grid[r-2, c] == color:
                        possible_colors.remove(color)
                        continue
                    if c >= 2 and grid[r, c-1] == color and grid[r, c-2] == color:
                        possible_colors.remove(color)
                        continue
                    grid[r, c] = color
                    break
                else: # If no color works, regenerate from scratch (rare)
                    return self._generate_initial_blocks()

        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                x, y = self._grid_to_pixel(r, c)
                self.blocks.append({
                    'r': r, 'c': c,
                    'color': grid[r, c],
                    'x': x, 'y': y,
                    'target_x': x, 'target_y': y,
                    'scale': 1.0,
                    'alpha': 255
                })

    def step(self, action):
        movement, space_held, shift_held = action
        reward = 0
        terminated = False

        # --- Handle Input and Game Logic ---
        # Selection change (on button press, not hold)
        if space_held and not self.previous_space_held:
            self._select_next_block()
        if shift_held and not self.previous_shift_held:
            self._select_previous_block()
        
        self.previous_space_held = bool(space_held)
        self.previous_shift_held = bool(shift_held)

        # Movement action (attempt a swap)
        if movement != 0 and not self.game_over:
            reward, move_was_valid = self._attempt_move(movement)
            if move_was_valid:
                self.steps += 1
                self.score += reward

        # --- Check Termination Conditions ---
        if not self.game_over:
            if not self.blocks: # Win condition
                terminated = True
                self.game_over = True
                reward += 100
                self.win_message = "YOU WIN!"
            elif self.steps >= self.MAX_STEPS: # Lose condition
                terminated = True
                self.game_over = True
                reward = -100 # Penalty for losing
                self.win_message = "GAME OVER"

        # Update animations for smooth visuals
        self._update_animations()

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _attempt_move(self, movement):
        dr, dc = [(0,0), (-1,0), (1,0), (0,-1), (0,1)][movement]
        r1, c1 = self.selected_pos
        r2, c2 = r1 + dr, c1 + dc

        if not (0 <= r2 < self.GRID_SIZE and 0 <= c2 < self.GRID_SIZE):
            return 0, False # Invalid move: out of bounds

        block1 = self._get_block_at(r1, c1)
        block2 = self._get_block_at(r2, c2)
        if not block1 or not block2:
            return 0, False # Should not happen in normal play

        # Temporarily swap and check for matches
        block1['r'], block1['c'] = r2, c2
        block2['r'], block2['c'] = r1, c1
        
        matches = self._find_all_matches()
        
        if not matches:
            # No match, revert swap
            block1['r'], block1['c'] = r1, c1
            block2['r'], block2['c'] = r2, c2
            return 0, False # Invalid move: no match created

        # Valid move, make swap permanent
        block1['target_x'], block1['target_y'] = self._grid_to_pixel(r2, c2)
        block2['target_x'], block2['target_y'] = self._grid_to_pixel(r1, c1)
        self.selected_pos = [r2, c2] # Move selection with the block

        # --- Resolve Board ---
        total_cleared = 0
        chain = 1
        while matches:
            # Clear matched blocks
            cleared_this_turn = len(matches)
            total_cleared += cleared_this_turn * chain # Bonus for chains
            
            blocks_to_remove = {b for b in self.blocks if (b['r'], b['c']) in matches}
            for b in blocks_to_remove:
                self._create_particles(b['x'], b['y'], b['color'])
                self.blocks.remove(b)
            # sfx: block_clear.wav

            # Apply gravity
            self._apply_gravity()

            # Find new matches for chain reaction
            matches = self._find_all_matches()
            chain += 1

        return total_cleared, True

    def _find_all_matches(self):
        grid = self._get_grid_from_blocks()
        matched_coords = set()
        
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE - 2):
                color = grid[r, c]
                if color != 0:
                    if c + 2 < self.GRID_SIZE and grid[r, c+1] == color and grid[r, c+2] == color:
                        for i in range(c, self.GRID_SIZE):
                            if grid[r, i] == color: matched_coords.add((r, i))
                            else: break
        
        for c in range(self.GRID_SIZE):
            for r in range(self.GRID_SIZE - 2):
                color = grid[r, c]
                if color != 0:
                    if r + 2 < self.GRID_SIZE and grid[r+1, c] == color and grid[r+2, c] == color:
                        for i in range(r, self.GRID_SIZE):
                            if grid[i, c] == color: matched_coords.add((i, c))
                            else: break
        
        return matched_coords

    def _apply_gravity(self):
        for c in range(self.GRID_SIZE):
            col_blocks = sorted([b for b in self.blocks if b['c'] == c], key=lambda b: b['r'])
            for i, block in enumerate(reversed(col_blocks)):
                new_r = self.GRID_SIZE - 1 - i
                if block['r'] != new_r:
                    block['r'] = new_r
                    block['target_x'], block['target_y'] = self._grid_to_pixel(new_r, c)
                    # sfx: block_fall.wav

    def _update_animations(self):
        for b in self.blocks:
            b['x'] += (b['target_x'] - b['x']) * self.ANIMATION_SPEED
            b['y'] += (b['target_y'] - b['y']) * self.ANIMATION_SPEED
        
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # particle gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        grid_rect = pygame.Rect(self.GRID_ORIGIN[0], self.GRID_ORIGIN[1], self.GRID_WIDTH, self.GRID_WIDTH)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect, border_radius=10)

        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / p['max_life']))
            color = p['color']
            s = self.screen.copy()
            s.set_alpha(alpha)
            pygame.draw.rect(s, color, (p['x'], p['y'], p['size'], p['size']))
            self.screen.blit(s, (0,0), special_flags=pygame.BLEND_RGBA_ADD)

        for b in self.blocks:
            color = self.BLOCK_COLORS[b['color'] - 1]
            rect = pygame.Rect(
                b['x'] + self.CELL_SIZE * 0.05,
                b['y'] + self.CELL_SIZE * 0.05,
                self.CELL_SIZE * 0.9,
                self.CELL_SIZE * 0.9
            )
            pygame.draw.rect(self.screen, color, rect, border_radius=8)
            highlight_color = tuple(min(255, val + 40) for val in color)
            inner_rect = rect.inflate(-8, -8)
            pygame.draw.rect(self.screen, highlight_color, inner_rect, border_radius=6)

        if not self.game_over and self._get_block_at(*self.selected_pos):
            sel_x, sel_y = self._grid_to_pixel(*self.selected_pos)
            pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2
            thickness = int(2 + pulse * 3)
            sel_rect = pygame.Rect(sel_x, sel_y, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_SELECTION, sel_rect.inflate(4,4), thickness, border_radius=12)

    def _render_ui(self):
        moves_text = self.font.render(f"Moves: {self.steps} / {self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 20))
        score_text = self.font.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 50))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(self.win_message, True, self.COLOR_SELECTION)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _grid_to_pixel(self, r, c):
        return (self.GRID_ORIGIN[0] + c * self.CELL_SIZE, self.GRID_ORIGIN[1] + r * self.CELL_SIZE)

    def _get_block_at(self, r, c):
        for block in self.blocks:
            if block['r'] == r and block['c'] == c:
                return block
        return None
    
    def _get_grid_from_blocks(self):
        grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        for block in self.blocks:
            grid[block['r'], block['c']] = block['color']
        return grid

    def _select_next_block(self):
        if not self.blocks: return
        valid_blocks = sorted(self.blocks, key=lambda b: (b['r'], b['c']))
        try:
            current_index = [i for i, b in enumerate(valid_blocks) if b['r'] == self.selected_pos[0] and b['c'] == self.selected_pos[1]][0]
            next_index = (current_index + 1) % len(valid_blocks)
        except (IndexError, ValueError):
            next_index = 0
        self.selected_pos = [valid_blocks[next_index]['r'], valid_blocks[next_index]['c']]
        # sfx: select.wav

    def _select_previous_block(self):
        if not self.blocks: return
        valid_blocks = sorted(self.blocks, key=lambda b: (b['r'], b['c']))
        try:
            current_index = [i for i, b in enumerate(valid_blocks) if b['r'] == self.selected_pos[0] and b['c'] == self.selected_pos[1]][0]
            next_index = (current_index - 1 + len(valid_blocks)) % len(valid_blocks)
        except (IndexError, ValueError):
            next_index = 0
        self.selected_pos = [valid_blocks[next_index]['r'], valid_blocks[next_index]['c']]
        # sfx: select.wav

    def _create_particles(self, x, y, color_idx):
        color = self.BLOCK_COLORS[color_idx - 1]
        center_x, center_y = x + self.CELL_SIZE // 2, y + self.CELL_SIZE // 2
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            life = self.np_random.integers(20, 40)
            self.particles.append({
                'x': center_x, 'y': center_y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': life, 'max_life': life,
                'size': self.np_random.integers(3, 8),
                'color': color
            })

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # Switch to a real video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    
    pygame.display.set_caption("Block Puzzle Environment")
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    print("\n--- Manual Play Controls ---")
    print(GameEnv.user_guide)
    print("R: Reset environment")
    print("--------------------------\n")
    
    while running:
        action = [0, 0, 0] # [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                
                if event.key == pygame.K_SPACE: action[1] = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1

        # Only step if an action is taken for this turn-based game
        if any(a != 0 for a in action):
            obs, reward, terminated, truncated, info = env.step(action)
            
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.0f}, Score: {info['score']:.0f}")

            if terminated or truncated:
                print(f"Episode finished. Final Score: {info['score']:.0f}")
                
                # Render final state
                surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                display_screen.blit(surf, (0, 0))
                pygame.display.flip()
                
                pygame.time.wait(2000)
                obs, info = env.reset()
        else: # if no action, just render the current state
             obs = env._get_observation()


        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30)

    env.close()