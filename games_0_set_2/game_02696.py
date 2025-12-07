
# Generated: 2025-08-27T21:11:18.815916
# Source Brief: brief_02696.md
# Brief Index: 2696

        
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
        "Controls: Use arrow keys (Up, Down, Left, Right) to push all blocks in that direction."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist puzzle game. Push colored blocks into their matching goal zones within the move limit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 12, 8
        self.CELL_SIZE = 40
        self.GRID_WIDTH = self.GRID_COLS * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_ROWS * self.CELL_SIZE
        self.GRID_X_OFFSET = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y_OFFSET = (self.HEIGHT - self.GRID_HEIGHT) // 2
        
        self.NUM_BLOCKS = 10
        self.MAX_MOVES = 20
        self.SCRAMBLE_MOVES = 12 # How many random moves to make from solved state
        self.ANIMATION_STEPS = 8 # Number of frames for block movement animation

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.BLOCK_COLORS = [
            (255, 87, 87),    # Red
            (87, 255, 87),    # Green
            (87, 177, 255),   # Blue
            (255, 255, 87),   # Yellow
            (255, 87, 255),   # Magenta
            (87, 255, 255),   # Cyan
            (255, 165, 0),    # Orange
            (188, 143, 143),  # Rosy Brown
            (127, 255, 0),    # Chartreuse
            (218, 112, 214),  # Orchid
        ]
        self.GOAL_ALPHA = 60
        
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
        try:
            self.font_large = pygame.font.SysFont("Consolas", 24)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 30)

        # --- Game State ---
        self.blocks = []
        self.goals = []
        self.moves_left = 0
        self.score = 0
        self.game_over = False
        self.animation_state = [] # List of {'block', 'start_pos', 'end_pos'}
        self.animation_progress = 1.0 # 0.0 to 1.0
        self.particles = []

        # Initialize state variables
        self.reset()
        
        # --- Final Validation ---
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.moves_left = self.MAX_MOVES
        self.score = 0
        self.game_over = False
        self.animation_state = []
        self.animation_progress = 1.0
        self.particles = []

        # Generate unique, non-overlapping positions for goals
        all_positions = [(x, y) for x in range(self.GRID_COLS) for y in range(self.GRID_ROWS)]
        self.np_random.shuffle(all_positions)
        goal_positions = all_positions[:self.NUM_BLOCKS]

        self.goals = []
        self.blocks = []
        for i in range(self.NUM_BLOCKS):
            color = self.BLOCK_COLORS[i]
            pos = goal_positions[i]
            self.goals.append({'pos': pos, 'color': color})
            self.blocks.append({
                'id': i,
                'pos': pos, 
                'color': color,
                'is_locked': False
            })

        # Scramble the board from the solved state to guarantee solvability
        self._scramble_board()
        
        # After scrambling, lock any blocks that ended up in their goals
        for block in self.blocks:
            goal = self.goals[block['id']]
            if block['pos'] == goal['pos']:
                block['is_locked'] = True
        
        return self._get_observation(), self._get_info()
    
    def _scramble_board(self):
        for _ in range(self.SCRAMBLE_MOVES):
            # Choose a random push direction
            direction = self.np_random.integers(1, 5) 
            self._execute_push(direction, is_scramble=True)
            # Reset lock state during scramble
            for block in self.blocks:
                block['is_locked'] = False

    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        terminated = False

        if movement != 0: # A push action was taken
            # If an animation is in progress, snap it to the end
            if self.animation_progress < 1.0:
                self._snap_animation_to_end()
            
            self.moves_left -= 1
            
            # --- Calculate pre-move state for reward ---
            pre_move_distances = {
                b['id']: self._manhattan_distance(b['pos'], self.goals[b['id']]['pos']) 
                for b in self.blocks if not b['is_locked']
            }

            # --- Execute the push ---
            moved_blocks = self._execute_push(movement)

            # --- Calculate post-move state and reward ---
            for block in self.blocks:
                if block['is_locked']: continue

                goal = self.goals[block['id']]
                
                # Check for new locks
                if block['pos'] == goal['pos']:
                    block['is_locked'] = True
                    reward += 10
                    # Sound effect placeholder: # sfx_lock_in.play()
                    self._create_particles(block['pos'], block['color'])
                
                # Distance-based reward
                post_move_distance = self._manhattan_distance(block['pos'], goal['pos'])
                prev_dist = pre_move_distances.get(block['id'], post_move_distance)
                if post_move_distance < prev_dist:
                    reward += 1 # Moved closer
                elif post_move_distance > prev_dist:
                    reward -= 1 # Moved further
            
            # --- Prepare animation ---
            self.animation_state = moved_blocks
            self.animation_progress = 0.0 if moved_blocks else 1.0

        # --- Check for termination ---
        num_locked = sum(1 for b in self.blocks if b['is_locked'])
        if num_locked == self.NUM_BLOCKS:
            terminated = True
            self.game_over = True
            reward += 100 # Win bonus
            # Sound effect placeholder: # sfx_win.play()
        elif self.moves_left <= 0:
            terminated = True
            self.game_over = True
            reward -= 50 # Loss penalty
            # Sound effect placeholder: # sfx_lose.play()

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _execute_push(self, direction, is_scramble=False):
        if direction not in [1, 2, 3, 4]: return []

        dx = {3: -1, 4: 1}.get(direction, 0)
        dy = {1: -1, 2: 1}.get(direction, 0)
        
        sort_key = lambda b: b['pos'][0]
        reverse = (direction == 4) # Right
        if direction in [1, 2]: # Up/Down
            sort_key = lambda b: b['pos'][1]
            reverse = (direction == 2) # Down
        
        sorted_blocks = sorted(self.blocks, key=sort_key, reverse=reverse)
        
        moved_blocks_info = []
        
        something_moved_in_pass = True
        while something_moved_in_pass:
            something_moved_in_pass = False
            occupied_positions = {b['pos'] for b in self.blocks}

            for block in sorted_blocks:
                if block['is_locked']: continue

                old_pos = block['pos']
                next_pos = (old_pos[0] + dx, old_pos[1] + dy)

                if (0 <= next_pos[0] < self.GRID_COLS and
                    0 <= next_pos[1] < self.GRID_ROWS and
                    next_pos not in occupied_positions):
                    
                    block['pos'] = next_pos
                    occupied_positions.remove(old_pos)
                    occupied_positions.add(next_pos)
                    something_moved_in_pass = True
                    
                    if not is_scramble:
                        found = False
                        for move_info in moved_blocks_info:
                            if move_info['block']['id'] == block['id']:
                                move_info['end_pos'] = next_pos
                                found = True
                                break
                        if not found:
                             moved_blocks_info.append({
                                'block': block, 
                                'start_pos': old_pos, 
                                'end_pos': next_pos
                            })
        return moved_blocks_info

    def _snap_animation_to_end(self):
        for anim in self.animation_state:
            block = next((b for b in self.blocks if b['id'] == anim['block']['id']), None)
            if block:
                block['pos'] = anim['end_pos']
        self.animation_progress = 1.0
        self.animation_state = []

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        if self.animation_progress < 1.0:
            self.animation_progress = min(1.0, self.animation_progress + 1.0 / self.ANIMATION_STEPS)
        self._update_particles()

        self._draw_grid()
        self._draw_goals()
        self._draw_blocks()
        self._draw_particles()

    def _draw_grid(self):
        for x in range(self.GRID_COLS + 1):
            px = self.GRID_X_OFFSET + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.GRID_Y_OFFSET), (px, self.GRID_Y_OFFSET + self.GRID_HEIGHT))
        for y in range(self.GRID_ROWS + 1):
            py = self.GRID_Y_OFFSET + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET, py), (self.GRID_X_OFFSET + self.GRID_WIDTH, py))
            
    def _draw_goals(self):
        for goal in self.goals:
            gx, gy = goal['pos']
            color = goal['color']
            rect = pygame.Rect(
                self.GRID_X_OFFSET + gx * self.CELL_SIZE,
                self.GRID_Y_OFFSET + gy * self.CELL_SIZE,
                self.CELL_SIZE, self.CELL_SIZE
            )
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            s.fill((*color, self.GOAL_ALPHA))
            self.screen.blit(s, rect.topleft)
            
    def _draw_blocks(self):
        current_block_positions = {b['id']: b['pos'] for b in self.blocks}
        
        if self.animation_progress < 1.0:
            eased_progress = 1 - (1 - self.animation_progress) ** 2
            for anim in self.animation_state:
                block_id = anim['block']['id']
                start_x, start_y = anim['start_pos']
                end_x, end_y = anim['end_pos']
                
                interp_x = start_x + (end_x - start_x) * eased_progress
                interp_y = start_y + (end_y - start_y) * eased_progress
                current_block_positions[block_id] = (interp_x, interp_y)

        for block in self.blocks:
            bx, by = current_block_positions[block['id']]
            color = block['color']
            
            px = self.GRID_X_OFFSET + bx * self.CELL_SIZE
            py = self.GRID_Y_OFFSET + by * self.CELL_SIZE
            
            rect = pygame.Rect(int(px), int(py), self.CELL_SIZE, self.CELL_SIZE)
            
            border_size = 4
            main_rect = pygame.Rect(rect.left + border_size, rect.top, rect.width - border_size, rect.height - border_size)
            shadow_rect = pygame.Rect(rect.left, rect.top + border_size, rect.width, rect.height - border_size)
            
            shadow_color = tuple(max(0, c - 50) for c in color)
            pygame.draw.rect(self.screen, shadow_color, shadow_rect, border_radius=4)
            pygame.draw.rect(self.screen, color, main_rect, border_radius=4)

            if block['is_locked']:
                center_x, center_y = rect.center
                pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, 8, (*self.COLOR_UI_TEXT, 150))
                pygame.gfxdraw.aacircle(self.screen, center_x, center_y, 8, (*self.COLOR_UI_TEXT, 200))

    def _render_ui(self):
        moves_text = f"Moves: {self.moves_left}"
        text_surf = self.font_large.render(moves_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (20, 15))

        solved_count = sum(1 for b in self.blocks if b['is_locked'])
        solved_text = f"Solved: {solved_count}/{self.NUM_BLOCKS}"
        text_surf = self.font_large.render(solved_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (self.WIDTH - text_surf.get_width() - 20, 15))

        if self.game_over:
            num_locked = sum(1 for b in self.blocks if b['is_locked'])
            msg, color = ("PUZZLE SOLVED!", (100, 255, 100)) if num_locked == self.NUM_BLOCKS else ("OUT OF MOVES", (255, 100, 100))
            
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            
            bg_rect = text_rect.inflate(20, 10)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((*self.COLOR_BG, 200))
            self.screen.blit(s, bg_rect)
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "moves_left": self.moves_left,
            "blocks_solved": sum(1 for b in self.blocks if b['is_locked']),
        }
        
    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _create_particles(self, grid_pos, color):
        center_x = self.GRID_X_OFFSET + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2
        center_y = self.GRID_Y_OFFSET + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'x': center_x, 'y': center_y,
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'size': self.np_random.uniform(2, 5),
                'life': self.np_random.integers(15, 31),
                'color': color,
            })

    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']; p['y'] += p['vy']
            p['life'] -= 1; p['size'] *= 0.95
        self.particles = [p for p in self.particles if p['life'] > 0 and p['size'] > 0.5]
    
    def _draw_particles(self):
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['x']), int(p['y'])), int(p['size']))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage for visualization
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Pusher")
    clock = pygame.time.Clock()
    running = True
    
    print("\n--- Block Pusher ---")
    print(GameEnv.game_description)
    print(GameEnv.user_guide)
    
    while running:
        action = [0, 0, 0] # Default action is no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    action = [0,0,0]
                elif event.key == pygame.K_q: running = False
        
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action[0]}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")
            if terminated:
                print("Game Over! Press 'R' to restart or 'Q' to quit.")

        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)
        
    env.close()