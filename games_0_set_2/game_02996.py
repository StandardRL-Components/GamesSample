
# Generated: 2025-08-27T22:03:17.752233
# Source Brief: brief_02996.md
# Brief Index: 2996

        
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
        "Controls: Use ← and → to cycle through blocks. Hold SPACE and use an arrow key (↑↓←→) to push the selected block. Solve in 5 moves for a bonus!"
    )

    game_description = (
        "A minimalist puzzle game. Push all colored blocks to their matching goals within the 25-move limit to win."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_W, self.SCREEN_H = 640, 400
        self.CELL_SIZE = 40
        self.GRID_W, self.GRID_H = self.SCREEN_W // self.CELL_SIZE, self.SCREEN_H // self.CELL_SIZE
        self.MAX_MOVES = 25
        self.WIN_MOVES_THRESHOLD = 20  # MAX_MOVES - 5
        self.NUM_BLOCKS = 5
        self.SCRAMBLE_MOVES = 30

        # --- Colors ---
        self.COLOR_BG = (26, 26, 26)
        self.COLOR_GRID = (51, 51, 51)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_SELECTION = (255, 255, 255)
        self.BLOCK_COLORS = [
            (255, 87, 87),    # Red
            (87, 255, 87),    # Green
            (87, 87, 255),    # Blue
            (255, 255, 87),   # Yellow
            (255, 87, 255),   # Magenta
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_H, self.SCREEN_W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_W, self.SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 48, bold=True)
        
        # --- State Variables ---
        self.blocks = []
        self.goals = []
        self.selected_block_idx = 0
        self.moves_remaining = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_status = "" # "" | "WIN" | "BONUS WIN" | "LOSE"

        self.reset()
        
        # self.validate_implementation() # Commented out for final submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_status = ""
        self.moves_remaining = self.MAX_MOVES
        self.selected_block_idx = 0
        
        self._generate_puzzle()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        action_taken = False

        if space_held and movement in [1, 2, 3, 4]: # Push action
            action_taken = True
            direction_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
            direction = direction_map[movement]
            
            block = self.blocks[self.selected_block_idx]
            goal = self.goals[self.selected_block_idx]
            
            old_dist = self._manhattan_distance(block['pos'], goal['pos'])
            
            # Placeholder for sound effect
            # sfx_push_start()
            
            moved = self._perform_push(self.selected_block_idx, direction)

            if moved:
                self.moves_remaining -= 1
                new_dist = self._manhattan_distance(block['pos'], goal['pos'])
                
                # Reward for distance change
                reward += (old_dist - new_dist)
                
                # Reward for landing on goal
                if new_dist == 0:
                    reward += 5
                    # Placeholder for sound effect
                    # sfx_block_on_goal()

        elif not space_held and movement in [3, 4]: # Select action
            if movement == 3: # Left
                self.selected_block_idx = (self.selected_block_idx - 1) % self.NUM_BLOCKS
            elif movement == 4: # Right
                self.selected_block_idx = (self.selected_block_idx + 1) % self.NUM_BLOCKS
            # Placeholder for sound effect
            # sfx_select_tick()

        self.steps += 1
        self.score += reward
        
        terminated, term_reward = self._check_termination()
        reward += term_reward
        self.score += term_reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _generate_puzzle(self):
        self.blocks = []
        self.goals = []
        
        occupied_positions = set()

        for i in range(self.NUM_BLOCKS):
            # Place goals
            while True:
                pos = (self.np_random.integers(1, self.GRID_W - 1), self.np_random.integers(1, self.GRID_H - 1))
                if pos not in occupied_positions:
                    self.goals.append({'pos': list(pos), 'color': self.BLOCK_COLORS[i]})
                    occupied_positions.add(pos)
                    break
            
            # Initially place blocks on goals
            self.blocks.append({'pos': list(self.goals[i]['pos']), 'color': self.BLOCK_COLORS[i]})
        
        # Scramble the puzzle by performing random pushes
        for _ in range(self.SCRAMBLE_MOVES):
            block_idx = self.np_random.integers(0, self.NUM_BLOCKS)
            direction_idx = self.np_random.integers(0, 4)
            direction = [(0, -1), (0, 1), (-1, 0), (1, 0)][direction_idx]
            self._perform_push(block_idx, direction)
            
    def _perform_push(self, block_idx, direction):
        block = self.blocks[block_idx]
        start_pos = list(block['pos'])
        current_pos = list(block['pos'])
        
        while True:
            next_pos = [current_pos[0] + direction[0], current_pos[1] + direction[1]]
            
            # Check wall collision
            if not (0 <= next_pos[0] < self.GRID_W and 0 <= next_pos[1] < self.GRID_H):
                break
                
            # Check other block collision
            if self._is_occupied(next_pos, ignored_block_idx=block_idx):
                break
            
            current_pos = next_pos
        
        block['pos'] = current_pos
        return current_pos != start_pos

    def _is_occupied(self, pos, ignored_block_idx=-1):
        for i, block in enumerate(self.blocks):
            if i == ignored_block_idx:
                continue
            if block['pos'][0] == pos[0] and block['pos'][1] == pos[1]:
                return True
        return False

    def _check_termination(self):
        if self.game_over:
            return True, 0

        all_on_goals = all(self._manhattan_distance(b['pos'], g['pos']) == 0 for b, g in zip(self.blocks, self.goals))

        if all_on_goals:
            self.game_over = True
            if self.moves_remaining >= self.WIN_MOVES_THRESHOLD:
                self.win_status = "BONUS WIN"
                # Placeholder for sound effect
                # sfx_win_bonus()
                return True, 50
            else:
                self.win_status = "WIN"
                # Placeholder for sound effect
                # sfx_win_normal()
                return True, 25
        
        if self.moves_remaining <= 0:
            self.game_over = True
            self.win_status = "LOSE"
            # Placeholder for sound effect
            # sfx_lose()
            return True, -50
            
        return False, 0

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

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
            "moves_remaining": self.moves_remaining,
            "win_status": self.win_status
        }
        
    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_W, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_H))
        for y in range(0, self.SCREEN_H, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_W, y))

        # Draw goals
        for goal in self.goals:
            x, y = goal['pos'][0] * self.CELL_SIZE, goal['pos'][1] * self.CELL_SIZE
            rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, goal['color'], rect, 3)

        # Draw blocks
        for i, block in enumerate(self.blocks):
            x, y = block['pos'][0] * self.CELL_SIZE, block['pos'][1] * self.CELL_SIZE
            rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
            
            # 3D effect
            border_color = tuple(max(0, c - 50) for c in block['color'])
            pygame.draw.rect(self.screen, border_color, rect)
            inner_rect = rect.inflate(-6, -6)
            pygame.draw.rect(self.screen, block['color'], inner_rect)

            # Draw selection highlight
            if i == self.selected_block_idx and not self.game_over:
                pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
                alpha = 100 + int(pulse * 155)
                
                # Use gfxdraw for anti-aliased circle
                sel_x, sel_y = rect.center
                sel_radius = int(self.CELL_SIZE * 0.6)
                pygame.gfxdraw.aacircle(self.screen, sel_x, sel_y, sel_radius, self.COLOR_SELECTION + (alpha,))
                pygame.gfxdraw.aacircle(self.screen, sel_x, sel_y, sel_radius-1, self.COLOR_SELECTION + (alpha,))


    def _render_ui(self):
        # Render moves remaining
        moves_text = self.font_main.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (10, 10))
        
        # Render game over status
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_W, self.SCREEN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win_status == "BONUS WIN":
                msg = "PERFECT!"
                color = self.BLOCK_COLORS[3] # Yellow
            elif self.win_status == "WIN":
                msg = "COMPLETE"
                color = self.BLOCK_COLORS[1] # Green
            else: # LOSE
                msg = "OUT OF MOVES"
                color = self.BLOCK_COLORS[0] # Red

            text_surf = self.font_big.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_W / 2, self.SCREEN_H / 2))
            self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Set up the display window
    pygame.display.set_caption(env.game_description)
    display_screen = pygame.display.set_mode((env.SCREEN_W, env.SCREEN_H))
    
    # Game loop
    running = True
    while running:
        action = [0, 0, 0] # Default no-op
        
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    terminated = False
                if event.key == pygame.K_q: # Quit on 'q'
                    running = False

        if not env.game_over:
            keys = pygame.key.get_pressed()
            
            movement = 0
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            # For this turn-based game, we only send an action if a key is pressed
            if movement != 0 or space_held != 0:
                action = [movement, space_held, shift_held]
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Moves: {info['moves_remaining']}, Terminated: {terminated}")
        
        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Since auto_advance is False, we control the "frame rate" of interaction
        env.clock.tick(15) # Limit player input rate

    env.close()