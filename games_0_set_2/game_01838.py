
# Generated: 2025-08-27T18:27:53.953697
# Source Brief: brief_01838.md
# Brief Index: 1838

        
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
        "Controls: Use arrow keys to push blocks. Press Space to restart the level."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist puzzle game. Push the colored blocks onto their matching goals before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        
        # Visuals
        self.font_large = pygame.font.Font(None, 60)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)

        self.COLOR_BG = (25, 35, 45)
        self.COLOR_GRID = (50, 65, 80)
        self.COLOR_UI_BG = (40, 55, 70)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_WIN_TEXT = (180, 255, 180)
        self.COLOR_LOSE_TEXT = (255, 180, 180)
        
        self.BLOCK_COLORS = [
            (255, 90, 90),   # Red
            (90, 150, 255),  # Blue
            (120, 255, 120), # Green
            (255, 220, 90),  # Yellow
            (255, 150, 90),  # Orange
            (180, 90, 255),  # Purple
        ]

        # Game state variables
        self.level = 0
        self.total_wins = 0
        self.grid_size = (0, 0)
        self.max_moves = 0
        self.blocks = []
        self.goals = []
        self.particles = []
        self.np_random = None

        # Initialize state
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.just_restarted = False
        
        self._generate_puzzle()
        
        self.satisfied_goals_last_step = set()

        return self._get_observation(), self._get_info()
    
    def _generate_puzzle(self):
        """Creates a solvable puzzle by starting with the solved state and scrambling it."""
        difficulty = self.level
        self.grid_size = (5 + difficulty // 5, 5 + difficulty // 5)
        num_blocks = min(len(self.BLOCK_COLORS), 2 + difficulty // 3)
        self.max_moves = 20 + (difficulty // 2) * 2
        self.moves_left = self.max_moves
        
        width, height = self.grid_size
        all_pos = [(x, y) for x in range(width) for y in range(height)]
        self.np_random.shuffle(all_pos)
        
        # Place goals and blocks in a solved state
        self.goals = []
        self.blocks = []
        colors = self.np_random.choice(len(self.BLOCK_COLORS), num_blocks, replace=False)
        
        for i in range(num_blocks):
            goal_pos = all_pos.pop()
            color = self.BLOCK_COLORS[colors[i]]
            self.goals.append({'pos': goal_pos, 'color': color})
            self.blocks.append({'pos': goal_pos, 'color': color, 'id': i})

        # Scramble the puzzle with random valid moves
        scramble_moves = num_blocks * 5 + difficulty * 2
        for _ in range(scramble_moves):
            self._scramble_move()

    def _scramble_move(self):
        """Performs a single random, valid 'un-push' to scramble the board."""
        movable_blocks = list(range(len(self.blocks)))
        self.np_random.shuffle(movable_blocks)
        
        for block_idx in movable_blocks:
            # Try to move this block in a random direction
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            self.np_random.shuffle(directions)
            for dx, dy in directions:
                if self._is_push_valid(block_idx, (dx, dy)):
                    self.blocks[block_idx]['pos'] = (
                        self.blocks[block_idx]['pos'][0] + dx,
                        self.blocks[block_idx]['pos'][1] + dy
                    )
                    return # One scramble move is enough

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        movement = action[0]
        space_held = action[1] == 1
        
        if space_held:
            self.game_over = True
            self.just_restarted = True
            reward = -10 # Penalty for restarting
            return self._get_observation(), reward, True, False, self._get_info()

        # Core game logic for a move
        self.moves_left -= 1
        reward -= 0.1 # Cost of making a move

        if movement > 0:
            direction = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
            self._attempt_push(direction)
        
        # Calculate rewards for block placement
        currently_satisfied = set()
        for i, block in enumerate(self.blocks):
            for goal in self.goals:
                if block['color'] == goal['color'] and block['pos'] == goal['pos']:
                    currently_satisfied.add(i)
                    break
        
        newly_satisfied = currently_satisfied - self.satisfied_goals_last_step
        for block_id in newly_satisfied:
            reward += 5.0
            block_pos = self.blocks[block_id]['pos']
            self._add_particles(block_pos, self.blocks[block_id]['color'])
        self.satisfied_goals_last_step = currently_satisfied

        # Check for termination conditions
        self.victory = len(currently_satisfied) == len(self.blocks)
        if self.victory:
            reward += 50.0
            self.game_over = True
            self.level += 1
            self.total_wins += 1
        elif self.moves_left <= 0:
            self.game_over = True

        terminated = self.game_over or self.steps >= 1000

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _attempt_push(self, direction):
        """Finds a random block that can be pushed in the given direction and moves it."""
        dx, dy = direction
        
        # Find all blocks that can be pushed in this direction
        pushable_block_indices = []
        for i, block in enumerate(self.blocks):
            if self._is_push_valid(i, direction):
                pushable_block_indices.append(i)
        
        if not pushable_block_indices:
            return # No block can be pushed

        # Select one block to push from the valid candidates
        block_to_push_idx = self.np_random.choice(pushable_block_indices)
        
        # Move the block
        block = self.blocks[block_to_push_idx]
        block['pos'] = (block['pos'][0] + dx, block['pos'][1] + dy)
        # sfx: block_push.wav
    
    def _is_push_valid(self, block_idx, direction):
        """Check if a block can be pushed in a direction."""
        block_pos = self.blocks[block_idx]['pos']
        target_pos = (block_pos[0] + direction[0], block_pos[1] + direction[1])
        
        # Check grid boundaries
        if not (0 <= target_pos[0] < self.grid_size[0] and 0 <= target_pos[1] < self.grid_size[1]):
            return False
        
        # Check collision with other blocks
        for i, other_block in enumerate(self.blocks):
            if i != block_idx and other_block['pos'] == target_pos:
                return False
                
        return True

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        """Renders the grid, goals, and blocks."""
        grid_w, grid_h = self.grid_size
        
        # Calculate cell size and offsets to center the grid
        available_w, available_h = 640, 340 # Leave 60px for UI
        cell_size = min(available_w // grid_w, available_h // grid_h)
        total_grid_pixel_w = grid_w * cell_size
        total_grid_pixel_h = grid_h * cell_size
        offset_x = (640 - total_grid_pixel_w) // 2
        offset_y = (400 - total_grid_pixel_h) // 2 + 20 # Push down for UI

        # Draw grid lines
        for x in range(grid_w + 1):
            start_pos = (offset_x + x * cell_size, offset_y)
            end_pos = (offset_x + x * cell_size, offset_y + total_grid_pixel_h)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
        for y in range(grid_h + 1):
            start_pos = (offset_x, offset_y + y * cell_size)
            end_pos = (offset_x + total_grid_pixel_w, offset_y + y * cell_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

        # Draw goals
        for goal in self.goals:
            gx, gy = goal['pos']
            px, py = offset_x + int(gx * cell_size), offset_y + int(gy * cell_size)
            goal_color = tuple(c * 0.5 for c in goal['color'])
            pygame.gfxdraw.filled_circle(self.screen, px + cell_size // 2, py + cell_size // 2, cell_size // 3, goal_color)
            pygame.gfxdraw.aacircle(self.screen, px + cell_size // 2, py + cell_size // 2, cell_size // 3, goal_color)

        # Draw blocks
        block_size = int(cell_size * 0.8)
        block_offset = (cell_size - block_size) // 2
        for block in self.blocks:
            bx, by = block['pos']
            px, py = offset_x + bx * cell_size + block_offset, offset_y + by * cell_size + block_offset
            
            shadow_color = tuple(c * 0.4 for c in block['color'])
            highlight_color = tuple(min(255, c * 1.5) for c in block['color'])
            
            rect = pygame.Rect(px, py, block_size, block_size)
            pygame.draw.rect(self.screen, shadow_color, rect.move(2, 2), border_radius=4)
            pygame.draw.rect(self.screen, block['color'], rect, border_radius=4)
            pygame.draw.rect(self.screen, highlight_color, rect.inflate(-block_size*0.7, -block_size*0.7).move(1,1), border_radius=2)
            
        # Update and draw particles
        self._update_and_draw_particles(offset_x, offset_y, cell_size)

    def _update_and_draw_particles(self, ox, oy, cell_size):
        active_particles = []
        for p in self.particles:
            p['life'] -= 1
            if p['life'] > 0:
                p['radius'] += p['growth']
                active_particles.append(p)
                
                px = ox + int(p['pos'][0] * cell_size) + cell_size // 2
                py = oy + int(p['pos'][1] * cell_size) + cell_size // 2
                
                alpha = int(255 * (p['life'] / p['max_life']))
                color = (*p['color'], alpha)
                
                temp_surf = pygame.Surface((int(p['radius'])*2, int(p['radius'])*2), pygame.SRCALPHA)
                pygame.gfxdraw.aacircle(temp_surf, int(p['radius']), int(p['radius']), int(p['radius']), color)
                self.screen.blit(temp_surf, (px - int(p['radius']), py - int(p['radius'])))

        self.particles = active_particles

    def _render_ui(self):
        # Draw UI background panel
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, 0, 640, 40))
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, 40), (640, 40))

        # Draw UI text
        level_text = self.font_medium.render(f"Level: {self.total_wins + 1}", True, self.COLOR_UI_TEXT)
        moves_text = self.font_medium.render(f"Moves: {self.moves_left}/{self.max_moves}", True, self.COLOR_UI_TEXT)
        
        self.screen.blit(level_text, (10, 8))
        self.screen.blit(moves_text, (630 - moves_text.get_width(), 8))
        
        # Draw game over message
        if self.game_over:
            overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.victory:
                # sfx: victory.wav
                msg = "PUZZLE SOLVED!"
                color = self.COLOR_WIN_TEXT
            elif self.just_restarted:
                msg = "LEVEL RESTARTED"
                color = self.COLOR_LOSE_TEXT
            else:
                # sfx: failure.wav
                msg = "OUT OF MOVES"
                color = self.COLOR_LOSE_TEXT
            
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(320, 200))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "wins": self.total_wins,
            "moves_left": self.moves_left,
            "is_victory": self.victory,
        }

    def _add_particles(self, pos, color):
        # sfx: score.wav
        for _ in range(15):
            life = self.np_random.integers(10, 30)
            self.particles.append({
                'pos': pos,
                'color': color,
                'radius': self.np_random.random() * 5,
                'growth': self.np_random.random() * 0.5 + 0.2,
                'life': life,
                'max_life': life
            })

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Game loop
    running = True
    while running:
        action = [0, 0, 0] # Default no-op action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r: # Manual reset
                    obs, info = env.reset()
                    terminated = False
        
        if not terminated:
            keys = pygame.key.get_pressed()
            # Movement
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            # Space
            if keys[pygame.K_SPACE]:
                action[1] = 1
            
            # Shift
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1

            if any(k > 0 for k in action):
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")

        # Pygame display update
        # The observation is already the rendered screen, so we just need to display it
        screen_to_show = pygame.transform.rotate(pygame.surfarray.make_surface(obs), -90)
        screen_to_show = pygame.transform.flip(screen_to_show, True, False)
        
        # Create a display if one doesn't exist
        try:
            display_surf = pygame.display.get_surface()
            if display_surf is None:
                 raise Exception
            display_surf.blit(screen_to_show, (0, 0))
        except Exception:
            display_surf = pygame.display.set_mode((640, 400))
            display_surf.blit(screen_to_show, (0, 0))

        pygame.display.flip()
        env.clock.tick(30) # Limit frame rate for human play

    env.close()