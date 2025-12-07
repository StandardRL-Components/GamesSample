
# Generated: 2025-08-28T02:27:24.047541
# Source Brief: brief_04454.md
# Brief Index: 4454

        
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
        "Controls: Use arrow keys (↑, ↓, ←, →) to push all blocks in a direction."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist puzzle game. Push entire rows and columns of blocks to slide them into their matching colored targets. Solve the puzzle in under 100 moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_SIZE = 5
    MAX_MOVES = 100
    MAX_STEPS = 1000

    # Colors (modern, high-contrast palette)
    COLOR_BG = (44, 62, 80)
    COLOR_GRID = (52, 73, 94)
    COLOR_TEXT = (236, 240, 241)
    COLOR_TEXT_SHADOW = (40, 40, 40)
    COLOR_WIN = (46, 204, 113)
    COLOR_LOSE = (231, 76, 60)

    BLOCK_COLORS = {
        "red": (231, 76, 60),
        "blue": (52, 152, 219),
        "green": (46, 204, 113),
        "yellow": (241, 196, 15),
    }

    # Pre-defined levels to ensure solvability and consistent difficulty
    LEVELS = [
        {
            "blocks": [("red", (1, 2)), ("blue", (3, 2))],
            "targets": [("red", (0, 4)), ("blue", (4, 4))],
        },
        {
            "blocks": [("red", (2, 1)), ("blue", (2, 3)), ("green", (1, 2))],
            "targets": [("red", (4, 1)), ("blue", (4, 3)), ("green", (0, 2))],
        },
        {
            "blocks": [("red", (0, 0)), ("blue", (4, 0)), ("green", (0, 4))],
            "targets": [("red", (4, 4)), ("blue", (2, 2)), ("green", (0, 0))],
        },
        {
            "blocks": [("red", (1, 1)), ("blue", (1, 3)), ("green", (3, 1)), ("yellow", (3, 3))],
            "targets": [("red", (0, 0)), ("blue", (0, 4)), ("green", (4, 0)), ("yellow", (4, 4))],
        },
    ]


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
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 60, bold=True)
        
        # Grid layout calculation
        self.grid_render_size = 300
        self.cell_size = self.grid_render_size // self.GRID_SIZE
        self.grid_render_size = self.cell_size * self.GRID_SIZE # Recalculate to avoid gaps
        self.grid_top_left = (
            (self.WIDTH - self.grid_render_size) // 2,
            (self.HEIGHT - self.grid_render_size) // 2,
        )

        # Initialize state variables
        self.blocks = []
        self.targets = []
        self.steps = 0
        self.moves_taken = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.current_level_idx = 0
        self.locked_blocks = set()

        # This will be called once in __init__
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.moves_taken = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        
        # Select a level using the seeded RNG
        self.current_level_idx = self.np_random.integers(0, len(self.LEVELS))
        level_data = self.LEVELS[self.current_level_idx]
        
        self.blocks = []
        self.targets = []
        self.locked_blocks.clear()

        for i, (color_name, pos) in enumerate(level_data["blocks"]):
            self.blocks.append({"id": i, "color": self.BLOCK_COLORS[color_name], "pos": list(pos)})
        
        for i, (color_name, pos) in enumerate(level_data["targets"]):
            self.targets.append({"id": i, "color": self.BLOCK_COLORS[color_name], "pos": list(pos)})
        
        self._update_locked_blocks()

        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0
        terminated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        if movement != 0: # Process a move
            self.moves_taken += 1
            reward -= 0.1 # Cost for making a move
            
            moved = self._perform_push(movement)
            # If no block could move, don't penalize the move count
            if not moved:
                self.moves_taken -= 1
                reward += 0.1

            newly_locked_count = self._update_locked_blocks()
            reward += newly_locked_count * 1.0

        self.steps += 1
        
        # Check for termination conditions
        if len(self.locked_blocks) == len(self.blocks):
            self.game_over = True
            self.win_state = True
            terminated = True
            reward += 100.0 # Win bonus
        elif self.moves_taken >= self.MAX_MOVES:
            self.game_over = True
            self.win_state = False
            terminated = True
            reward -= 100.0 # Lose penalty
        elif self.steps >= self.MAX_STEPS:
            # Game truncated if it runs too long without solving
            self.game_over = True
            self.win_state = False
            terminated = True


        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False as per brief
            self._get_info()
        )

    def _perform_push(self, direction):
        # direction: 1=up, 2=down, 3=left, 4=right
        moved_any_block = False
        
        # Create a map for quick lookups of block presence and lock status
        grid_map = {}
        for block in self.blocks:
            grid_map[tuple(block['pos'])] = block['id'] in self.locked_blocks

        # Up
        if direction == 1:
            for x in range(self.GRID_SIZE):
                for y in range(1, self.GRID_SIZE):
                    if tuple([x, y]) in grid_map and not grid_map[tuple([x, y])]:
                        target_y = y
                        while target_y > 0 and tuple([x, target_y - 1]) not in grid_map:
                            target_y -= 1
                        if target_y != y:
                            # Move block in logic
                            block = self._get_block_at((x, y))
                            block['pos'][1] = target_y
                            # Update map for subsequent checks in this push
                            del grid_map[tuple([x, y])]
                            grid_map[tuple([x, target_y])] = False
                            moved_any_block = True
        # Down
        elif direction == 2:
            for x in range(self.GRID_SIZE):
                for y in reversed(range(self.GRID_SIZE - 1)):
                    if tuple([x, y]) in grid_map and not grid_map[tuple([x, y])]:
                        target_y = y
                        while target_y < self.GRID_SIZE - 1 and tuple([x, target_y + 1]) not in grid_map:
                            target_y += 1
                        if target_y != y:
                            block = self._get_block_at((x, y))
                            block['pos'][1] = target_y
                            del grid_map[tuple([x, y])]
                            grid_map[tuple([x, target_y])] = False
                            moved_any_block = True
        # Left
        elif direction == 3:
            for y in range(self.GRID_SIZE):
                for x in range(1, self.GRID_SIZE):
                    if tuple([x, y]) in grid_map and not grid_map[tuple([x, y])]:
                        target_x = x
                        while target_x > 0 and tuple([target_x - 1, y]) not in grid_map:
                            target_x -= 1
                        if target_x != x:
                            block = self._get_block_at((x, y))
                            block['pos'][0] = target_x
                            del grid_map[tuple([x, y])]
                            grid_map[tuple([target_x, y])] = False
                            moved_any_block = True
        # Right
        elif direction == 4:
            for y in range(self.GRID_SIZE):
                for x in reversed(range(self.GRID_SIZE - 1)):
                    if tuple([x, y]) in grid_map and not grid_map[tuple([x, y])]:
                        target_x = x
                        while target_x < self.GRID_SIZE - 1 and tuple([target_x + 1, y]) not in grid_map:
                            target_x += 1
                        if target_x != x:
                            block = self._get_block_at((x, y))
                            block['pos'][0] = target_x
                            del grid_map[tuple([x, y])]
                            grid_map[tuple([target_x, y])] = False
                            moved_any_block = True
        
        return moved_any_block

    def _update_locked_blocks(self):
        newly_locked = 0
        for block in self.blocks:
            if block['id'] in self.locked_blocks:
                continue
            for target in self.targets:
                if block['color'] == target['color'] and tuple(block['pos']) == tuple(target['pos']):
                    if block['id'] not in self.locked_blocks:
                        self.locked_blocks.add(block['id'])
                        newly_locked += 1
                        # Sound effect placeholder: # sfx_lock_in
        return newly_locked

    def _get_block_at(self, pos):
        for block in self.blocks:
            if tuple(block['pos']) == tuple(pos):
                return block
        return None

    def _pos_to_pixels(self, pos):
        px = self.grid_top_left[0] + pos[0] * self.cell_size
        py = self.grid_top_left[1] + pos[1] * self.cell_size
        return int(px), int(py)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_pos = (self.grid_top_left[0] + i * self.cell_size, self.grid_top_left[1])
            end_pos = (self.grid_top_left[0] + i * self.cell_size, self.grid_top_left[1] + self.grid_render_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 2)
            # Horizontal
            start_pos = (self.grid_top_left[0], self.grid_top_left[1] + i * self.cell_size)
            end_pos = (self.grid_top_left[0] + self.grid_render_size, self.grid_top_left[1] + i * self.cell_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 2)
        
        # Draw targets
        for target in self.targets:
            px, py = self._pos_to_pixels(target['pos'])
            rect = pygame.Rect(px, py, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, target['color'], rect, width=4)

        # Draw blocks
        for block in self.blocks:
            px, py = self._pos_to_pixels(block['pos'])
            is_locked = block['id'] in self.locked_blocks
            
            # If locked, make it look "seated" in the target
            inset = self.cell_size // 6 if is_locked else 4
            
            rect = pygame.Rect(px + inset, py + inset, self.cell_size - inset * 2, self.cell_size - inset * 2)
            pygame.draw.rect(self.screen, block['color'], rect, border_radius=4)
            
            if is_locked:
                # Draw a checkmark-like symbol for locked blocks
                center_x, center_y = rect.center
                p1 = (center_x - 10, center_y)
                p2 = (center_x - 2, center_y + 8)
                p3 = (center_x + 10, center_y - 8)
                pygame.draw.lines(self.screen, self.COLOR_BG, False, [p1, p2, p3], 4)

    def _render_ui(self):
        # Render move counter
        moves_text = f"Moves: {self.moves_taken} / {self.MAX_MOVES}"
        self._draw_text(moves_text, (20, 20), self.font_main, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        # Render score
        score_text = f"Score: {self.score:.1f}"
        self._draw_text(score_text, (self.WIDTH - 180, 20), self.font_main, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        # Render game over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win_state:
                msg = "YOU WIN!"
                color = self.COLOR_WIN
            else:
                msg = "TOO MANY MOVES"
                color = self.COLOR_LOSE
            
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(text_surf, text_rect)

    def _draw_text(self, text, pos, font, color, shadow_color):
        text_surf_shadow = font.render(text, True, shadow_color)
        self.screen.blit(text_surf_shadow, (pos[0] + 2, pos[1] + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_taken": self.moves_taken,
            "blocks_locked": len(self.locked_blocks),
            "total_blocks": len(self.blocks)
        }
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
        print("Running implementation validation...")
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


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It's a simple example of how to use the environment
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Block Pusher")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print("      BLOCK PUSHER - DEMO")
    print("="*30)
    print(env.game_description)
    print(env.user_guide)
    print("Press 'R' to reset the level.")
    print("Press 'Q' or close the window to quit.")
    print("="*30 + "\n")


    while running:
        action = np.array([0, 0, 0]) # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Level Reset ---")
                
                # Map keys to actions
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
        
        # Only step if an action was taken
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"Move: {info['moves_taken']}, Step Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Locked: {info['blocks_locked']}/{info['total_blocks']}")

            if terminated:
                print("\n--- GAME OVER ---")
                print(f"Final Score: {info['score']:.2f}")
                print("Press 'R' to play again or 'Q' to quit.")


        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit frame rate
        
    env.close()