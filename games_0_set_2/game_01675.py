
# Generated: 2025-08-28T02:20:00.613846
# Source Brief: brief_01675.md
# Brief Index: 1675

        
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
        "Controls: Use arrow keys to move your blue block. Pushing a block will push all connected blocks in that line."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A grid-based puzzle game. Push the red target block to the green exit square before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.MAX_MOVES = 20
        self.MAX_STEPS = 1000

        # Visuals
        self.CELL_SIZE = 36
        self.GRID_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_X_OFFSET = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y_OFFSET = (self.HEIGHT - self.GRID_HEIGHT) // 2
        
        # Colors
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_GRID = (50, 60, 70)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_GLOW = (150, 200, 255)
        self.COLOR_TARGET = (255, 70, 70)
        self.COLOR_EXIT = (70, 255, 70)
        self.COLOR_TEXT = (230, 230, 230)
        self.BLOCK_COLORS = [
            (255, 190, 0), (200, 0, 200), (0, 200, 200),
            (255, 120, 0), (150, 50, 255)
        ]
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_level = pygame.font.Font(None, 28)

        # Persistent state
        self.completed_levels = 0
        
        # Initialize state variables
        self.grid = []
        self.player_pos = (0, 0)
        self.target_pos = (0, 0)
        self.exit_pos = (0, 0)
        self.blocks = []
        self.last_action_effects = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = 0
        self.level = 1
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = self.MAX_MOVES
        self.level = self.completed_levels + 1
        self.last_action_effects = []

        self._generate_level()
        
        self.initial_target_dist = self._manhattan_distance(self.target_pos, self.exit_pos)

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.grid = [[None for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        
        num_blocks = 3 + (self.level - 1) // 5
        
        possible_coords = [(x, y) for x in range(self.GRID_SIZE) for y in range(self.GRID_SIZE)]
        self.np_random.shuffle(possible_coords)

        self.exit_pos = possible_coords.pop()
        self.target_pos = possible_coords.pop()
        self.player_pos = possible_coords.pop()
        
        self.grid[self.target_pos[1]][self.target_pos[0]] = "target"
        self.grid[self.player_pos[1]][self.player_pos[0]] = "player"

        self.blocks = []
        for i in range(num_blocks):
            if not possible_coords: break
            pos = possible_coords.pop()
            color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
            block_id = f"block_{i}"
            self.blocks.append({"pos": pos, "color": color, "id": block_id})
            self.grid[pos[1]][pos[0]] = block_id

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        self.last_action_effects = []

        old_target_dist = self._manhattan_distance(self.target_pos, self.exit_pos)

        if movement > 0: # Any action other than no-op
            self.moves_remaining -= 1
            reward -= 0.1 # Cost of moving
            
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
            
            self._handle_push(dx, dy)

        new_target_dist = self._manhattan_distance(self.target_pos, self.exit_pos)
        reward += (old_target_dist - new_target_dist) # +1 if closer, -1 if further

        self.score += reward
        self.steps += 1
        
        terminated = self._check_termination()

        if terminated:
            if self.target_pos == self.exit_pos:
                win_reward = 100
                reward += win_reward
                self.score += win_reward
                self.completed_levels += 1
                # placeholder: sfx_win.play()
            else:
                loss_penalty = -100
                reward += loss_penalty
                self.score += loss_penalty
                # placeholder: sfx_lose.play()
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_push(self, dx, dy):
        px, py = self.player_pos
        chain = []
        next_x, next_y = px + dx, py + dy
        
        # Identify the contiguous chain of blocks
        while 0 <= next_x < self.GRID_SIZE and 0 <= next_y < self.GRID_SIZE:
            if self.grid[next_y][next_x] is not None and self.grid[next_y][next_x] != "player":
                chain.append((next_x, next_y))
                next_x += dx
                next_y += dy
            else:
                break
        
        # Check if the entire chain can be pushed
        end_of_chain_x, end_of_chain_y = next_x, next_y
        if 0 <= end_of_chain_x < self.GRID_SIZE and 0 <= end_of_chain_y < self.GRID_SIZE and self.grid[end_of_chain_y][end_of_chain_x] is None:
            # Move all blocks in the chain, starting from the back
            for i in range(len(chain) - 1, -1, -1):
                block_x, block_y = chain[i]
                new_pos = (block_x + dx, block_y + dy)
                block_id = self.grid[block_y][block_x]

                self.grid[new_pos[1]][new_pos[0]] = block_id
                self.grid[block_y][block_x] = None

                # Update state and record visual effect
                if block_id == "target":
                    self.target_pos = new_pos
                    self.last_action_effects.append(((block_x, block_y), new_pos, self.COLOR_TARGET))
                else:
                    for b in self.blocks:
                        if b["id"] == block_id:
                            b["pos"] = new_pos
                            self.last_action_effects.append(((block_x, block_y), new_pos, b["color"]))
                            break
            
            # Move player
            new_player_pos = (px + dx, py + dy)
            self.grid[py][px] = None
            self.grid[new_player_pos[1]][new_player_pos[0]] = "player"
            self.last_action_effects.append((self.player_pos, new_player_pos, self.COLOR_PLAYER))
            self.player_pos = new_player_pos
            # placeholder: sfx_push.play()
        else:
            # Check if player can move into an empty space
            nx, ny = px + dx, py + dy
            if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and self.grid[ny][nx] is None:
                new_player_pos = (nx, ny)
                self.grid[py][px] = None
                self.grid[new_player_pos[1]][new_player_pos[0]] = "player"
                self.last_action_effects.append((self.player_pos, new_player_pos, self.COLOR_PLAYER))
                self.player_pos = new_player_pos
                # placeholder: sfx_move.play()
            else:
                # placeholder: sfx_bump.play()
                pass # Bumped into wall or unmovable chain

    def _check_termination(self):
        if self.target_pos == self.exit_pos:
            return True
        if self.moves_remaining <= 0:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_trails()
        self._render_grid()
        self._render_entities()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_trails(self):
        for start_pos, end_pos, color in self.last_action_effects:
            start_px = self._grid_to_pixel(start_pos)
            end_px = self._grid_to_pixel(end_pos)
            
            trail_color = tuple(c * 0.5 for c in color)
            
            rect = pygame.Rect(
                min(start_px[0], end_px[0]),
                min(start_px[1], end_px[1]),
                self.CELL_SIZE if start_px[1] == end_px[1] else abs(start_px[0] - end_px[0]) + self.CELL_SIZE,
                self.CELL_SIZE if start_px[0] == end_px[0] else abs(start_px[1] - end_px[1]) + self.CELL_SIZE,
            )
            pygame.draw.rect(self.screen, trail_color, rect, border_radius=8)

    def _render_grid(self):
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            start_pos = (self.GRID_X_OFFSET + i * self.CELL_SIZE, self.GRID_Y_OFFSET)
            end_pos = (self.GRID_X_OFFSET + i * self.CELL_SIZE, self.GRID_Y_OFFSET + self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            # Horizontal lines
            start_pos = (self.GRID_X_OFFSET, self.GRID_Y_OFFSET + i * self.CELL_SIZE)
            end_pos = (self.GRID_X_OFFSET + self.GRID_WIDTH, self.GRID_Y_OFFSET + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
    
    def _render_entities(self):
        # Exit
        exit_px, exit_py = self._grid_to_pixel(self.exit_pos)
        exit_rect = pygame.Rect(exit_px, exit_py, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect, border_radius=8)

        # Blocks
        for block in self.blocks:
            px, py = self._grid_to_pixel(block["pos"])
            rect = pygame.Rect(px + 3, py + 3, self.CELL_SIZE - 6, self.CELL_SIZE - 6)
            pygame.draw.rect(self.screen, block["color"], rect, border_radius=6)

        # Target
        target_px, target_py = self._grid_to_pixel(self.target_pos)
        target_rect = pygame.Rect(target_px + 3, target_py + 3, self.CELL_SIZE - 6, self.CELL_SIZE - 6)
        pygame.draw.rect(self.screen, self.COLOR_TARGET, target_rect, border_radius=6)
        
        # Player
        player_px, player_py = self._grid_to_pixel(self.player_pos)
        # Glow effect
        glow_center = (player_px + self.CELL_SIZE // 2, player_py + self.CELL_SIZE // 2)
        pygame.gfxdraw.filled_circle(self.screen, glow_center[0], glow_center[1], self.CELL_SIZE // 2, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aacircle(self.screen, glow_center[0], glow_center[1], self.CELL_SIZE // 2, self.COLOR_PLAYER_GLOW)
        # Player block
        player_rect = pygame.Rect(player_px + 3, player_py + 3, self.CELL_SIZE - 6, self.CELL_SIZE - 6)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=6)

    def _render_ui(self):
        moves_text = self.font_ui.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 20))

        level_text = self.font_level.render(f"Level: {self.level}", True, self.COLOR_TEXT)
        level_rect = level_text.get_rect(topright=(self.WIDTH - 20, 20))
        self.screen.blit(level_text, level_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "moves_remaining": self.moves_remaining,
            "completed_levels": self.completed_levels,
        }

    def _grid_to_pixel(self, grid_pos):
        px = self.GRID_X_OFFSET + grid_pos[0] * self.CELL_SIZE
        py = self.GRID_Y_OFFSET + grid_pos[1] * self.CELL_SIZE
        return int(px), int(py)

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def close(self):
        pygame.font.quit()
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
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Pusher")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = np.array([0, 0, 0]) # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                elif event.key == pygame.K_ESCAPE:
                    running = False
        
        # Only step if an action was taken
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")
            if terminated:
                print("Game Over! Press 'R' to restart.")

        # Update the display
        draw_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(draw_surface, (0, 0))
        pygame.display.flip()
        
        clock.tick(30)
        
    env.close()