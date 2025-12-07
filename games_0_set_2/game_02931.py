# Generated: 2025-08-28T06:26:59.815718
# Source Brief: brief_02931.md
# Brief Index: 2931

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A grid-based puzzle game where the player moves colored blocks to their target locations.

    The player selects a block using the 'space' and 'shift' keys and moves it with the
    arrow keys. A block slides in the chosen direction until it hits another block or the
    edge of the grid. The goal is to get all blocks onto their corresponding targets
    within a limited number of moves.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Use Space/Shift to cycle selection. Use ←↑→↓ to slide the selected block."
    )

    # Short, user-facing description of the game
    game_description = (
        "A minimalist puzzle game. Slide colored blocks onto their matching targets before you run out of moves."
    )

    # Frames only advance when an action is received
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and grid dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_COLS = 10
        self.GRID_ROWS = 6
        self.GRID_MARGIN_X = (self.SCREEN_WIDTH - self.GRID_COLS * 60) // 2
        self.GRID_MARGIN_Y = (self.SCREEN_HEIGHT - self.GRID_ROWS * 60) // 2 + 20
        self.CELL_SIZE = 60

        # Colors
        self.COLOR_BG = (25, 28, 32)
        self.COLOR_GRID = (45, 50, 56)
        self.COLOR_UI_BG = (35, 38, 43)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TEXT_SUCCESS = (100, 255, 100)
        self.COLOR_TEXT_FAIL = (255, 100, 100)
        self.COLOR_SELECT = (255, 255, 255)
        self.BLOCK_COLORS = [
            (255, 95, 87),   # Red
            (87, 184, 255),  # Blue
            (87, 255, 150),  # Green
            (255, 239, 87),  # Yellow
            (200, 100, 255)  # Purple
        ]

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 72)
        
        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = 0
        self.max_moves = 0
        self.blocks = []
        self.selected_block_idx = 0
        self.last_action_was_move = False
        self.game_outcome_message = ""

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.selected_block_idx = 0
        self.last_action_was_move = False
        self.game_outcome_message = ""

        # Puzzle parameters
        num_blocks = self.np_random.integers(3, len(self.BLOCK_COLORS) + 1)
        self.max_moves = 25 + (num_blocks - 3) * 10
        self.moves_left = self.max_moves

        self._generate_puzzle(num_blocks)
        
        return self._get_observation(), self._get_info()

    def _generate_puzzle(self, num_blocks):
        """Generates a solvable puzzle by starting with the solution and shuffling."""
        all_positions = [(x, y) for x in range(self.GRID_COLS) for y in range(self.GRID_ROWS)]
        self.np_random.shuffle(all_positions)

        # Place targets
        target_positions = all_positions[:num_blocks]
        
        self.blocks = []
        for i in range(num_blocks):
            self.blocks.append({
                "pos": target_positions[i],
                "target": target_positions[i],
                "color": self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)],
            })
        
        # Shuffle blocks from solved state
        shuffle_steps = num_blocks * 5
        for _ in range(shuffle_steps):
            block_idx = self.np_random.integers(0, len(self.blocks))
            direction = self.np_random.integers(1, 5) # 1-4 for up/down/left/right
            
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[direction]
            
            current_pos = self.blocks[block_idx]["pos"]
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)
            
            # Check if move is valid (within bounds and not into another block)
            if 0 <= next_pos[0] < self.GRID_COLS and 0 <= next_pos[1] < self.GRID_ROWS:
                is_occupied = any(b["pos"] == next_pos for b in self.blocks)
                if not is_occupied:
                    self.blocks[block_idx]["pos"] = next_pos

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        self.last_action_was_move = False
        
        # 1. Handle selection change
        if shift_held:
            self.selected_block_idx = (self.selected_block_idx - 1 + len(self.blocks)) % len(self.blocks)
        elif space_held:
            self.selected_block_idx = (self.selected_block_idx + 1) % len(self.blocks)
        
        # 2. Handle movement
        if movement != 0:
            self.last_action_was_move = True
            self.moves_left -= 1
            reward -= 0.1  # Cost for making a move
            
            selected_block = self.blocks[self.selected_block_idx]
            was_on_target = selected_block["pos"] == selected_block["target"]
            
            # Slide logic
            self._slide_block(selected_block, movement)
            
            is_on_target = selected_block["pos"] == selected_block["target"]
            
            if is_on_target and not was_on_target:
                reward += 1.0  # Reward for placing a block correctly
                # Sound effect: block_placed.wav

        # 3. Check for termination conditions
        terminated = False
        all_on_target = all(b["pos"] == b["target"] for b in self.blocks)
        
        if all_on_target:
            reward += 100.0
            terminated = True
            self.game_over = True
            self.game_outcome_message = "PUZZLE SOLVED!"
            # Sound effect: puzzle_win.wav
        elif self.moves_left <= 0:
            reward -= 100.0
            terminated = True
            self.game_over = True
            self.game_outcome_message = "OUT OF MOVES"
            # Sound effect: puzzle_fail.wav

        self.score += reward
        self.steps += 1
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _slide_block(self, block, direction):
        """Slides a block in a direction until it hits an obstacle."""
        dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[direction]
        
        current_pos = block["pos"]
        while True:
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)
            
            # Check for wall collision
            if not (0 <= next_pos[0] < self.GRID_COLS and 0 <= next_pos[1] < self.GRID_ROWS):
                break
            
            # Check for block collision
            is_occupied = any(b["pos"] == next_pos for b in self.blocks if b is not block)
            if is_occupied:
                break
                
            current_pos = next_pos
        
        if block["pos"] != current_pos:
            # Sound effect: block_slide.wav
            block["pos"] = current_pos

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_COLS + 1):
            px = self.GRID_MARGIN_X + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.GRID_MARGIN_Y), (px, self.GRID_MARGIN_Y + self.GRID_ROWS * self.CELL_SIZE), 1)
        for y in range(self.GRID_ROWS + 1):
            py = self.GRID_MARGIN_Y + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_MARGIN_X, py), (self.GRID_MARGIN_X + self.GRID_COLS * self.CELL_SIZE, py), 1)

        # Draw targets
        for block in self.blocks:
            target_x, target_y = block["target"]
            px = self.GRID_MARGIN_X + target_x * self.CELL_SIZE
            py = self.GRID_MARGIN_Y + target_y * self.CELL_SIZE
            
            # Desaturate and lighten the block color for the target
            r, g, b = block["color"]
            h, s, v, _ = pygame.Color(r,g,b).hsva
            target_color = pygame.Color(0,0,0)
            target_color.hsva = (h, s * 0.4, v * 0.6, 100)
            
            pygame.draw.rect(self.screen, target_color, (px, py, self.CELL_SIZE, self.CELL_SIZE))
            pygame.draw.rect(self.screen, self.COLOR_GRID, (px + 4, py + 4, self.CELL_SIZE - 8, self.CELL_SIZE - 8), 2)

        # Draw blocks
        if self.blocks:
            for i, block in enumerate(self.blocks):
                block_x, block_y = block["pos"]
                px = self.GRID_MARGIN_X + block_x * self.CELL_SIZE
                py = self.GRID_MARGIN_Y + block_y * self.CELL_SIZE
                
                # Draw shadow
                shadow_offset = 3
                shadow_color = (0,0,0,50)
                shadow_rect = pygame.Rect(px + shadow_offset, py + shadow_offset, self.CELL_SIZE, self.CELL_SIZE)
                shape_surf = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
                pygame.draw.rect(shape_surf, shadow_color, (0, 0, self.CELL_SIZE, self.CELL_SIZE), border_radius=6)
                self.screen.blit(shape_surf, shadow_rect)

                # Draw block
                pygame.draw.rect(self.screen, block["color"], (px, py, self.CELL_SIZE, self.CELL_SIZE), border_radius=6)

                # Draw selection highlight
                if i == self.selected_block_idx:
                    pulse = (math.sin(self.steps * 0.2) + 1) / 2  # 0 to 1
                    thickness = 2 + int(pulse * 2)
                    pygame.draw.rect(self.screen, self.COLOR_SELECT, (px, py, self.CELL_SIZE, self.CELL_SIZE), thickness, border_radius=6)

    def _render_ui(self):
        # UI Background Panel
        ui_panel = pygame.Surface((self.SCREEN_WIDTH, 60), pygame.SRCALPHA)
        ui_panel.fill((*self.COLOR_UI_BG, 200))
        self.screen.blit(ui_panel, (0, 0))
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, 60), (self.SCREEN_WIDTH, 60))

        # Moves Left
        moves_text = self.font_main.render(f"Moves: {self.moves_left}/{self.max_moves}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 15))
        
        # Score
        score_text = self.font_main.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(right=self.SCREEN_WIDTH - 20, centery=30)
        self.screen.blit(score_text, score_rect)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            color = self.COLOR_TEXT_SUCCESS if "SOLVED" in self.game_outcome_message else self.COLOR_TEXT_FAIL
            message_surf = self.font_large.render(self.game_outcome_message, True, color)
            message_rect = message_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(message_surf, message_rect)

    def _get_info(self):
        is_solved = False
        if self.blocks:
            is_solved = all(b["pos"] == b["target"] for b in self.blocks)
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "is_solved": is_solved
        }

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    # Pygame setup for human play
    # Re-enable video driver for display
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Slider Puzzle")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    print(env.game_description)
    print(env.user_guide)

    while running:
        # Convert observation for display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.1f}")
            pygame.time.wait(3000) # Wait 3 seconds
            obs, info = env.reset()
            terminated = False
            continue

        action = [0, 0, 0] # no-op, release, release
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    terminated = False
                    continue # Skip stepping on reset
                
                # Map keys to MultiDiscrete action
                keys = pygame.key.get_pressed()
                
                # Movement
                if keys[pygame.K_UP]: action[0] = 1
                elif keys[pygame.K_DOWN]: action[0] = 2
                elif keys[pygame.K_LEFT]: action[0] = 3
                elif keys[pygame.K_RIGHT]: action[0] = 4
                
                # Buttons
                if keys[pygame.K_SPACE]: action[1] = 1
                if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
                
                # Since it's not auto-advancing, we step on any key press that's a valid action
                if action != [0,0,0]:
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.1f}, Terminated: {terminated}")
        
        clock.tick(30) # Limit frame rate for human play

    env.close()