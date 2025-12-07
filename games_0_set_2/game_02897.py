
# Generated: 2025-08-28T06:20:23.301546
# Source Brief: brief_02897.md
# Brief Index: 2897

        
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
        "Controls: Use arrow keys to move your avatar (white square) and push colored blocks into their matching goal zones."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro-style puzzle game. Push all the blocks to their goal locations before time runs out. Plan your moves carefully to avoid getting stuck!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = (16, 10)
        self.CELL_SIZE = 40
        self.MAX_STEPS = 5000
        self.TOTAL_TIME = 180.0
        self.TIME_PER_STEP = 0.25

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_OBSTACLE = (80, 80, 90)
        self.BLOCK_COLORS = {
            "red": (255, 70, 70),
            "blue": (70, 120, 255),
            "green": (70, 255, 120),
            "yellow": (255, 255, 70),
            "purple": (200, 70, 255),
        }
        self.GOAL_COLORS = {
            name: tuple(int(c * 0.4) for c in color)
            for name, color in self.BLOCK_COLORS.items()
        }
        self.COLOR_TEXT = (220, 220, 220)

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = 0.0
        self.current_level = 0
        self.player_pos = (0, 0)
        self.blocks = []
        self.obstacles = []
        self._level_definitions = self._define_levels()

        self.reset()
        self.validate_implementation()

    def _define_levels(self):
        return [
            # Level 1
            {
                "player": (1, 1),
                "blocks": [
                    {"pos": (3, 3), "type": "red"},
                    {"pos": (5, 5), "type": "blue"},
                ],
                "goals": [
                    {"pos": (7, 3), "type": "red"},
                    {"pos": (7, 5), "type": "blue"},
                ],
                "obstacles": [],
            },
            # Level 2
            {
                "player": (1, 4),
                "blocks": [
                    {"pos": (3, 2), "type": "red"},
                    {"pos": (3, 4), "type": "blue"},
                    {"pos": (3, 6), "type": "green"},
                ],
                "goals": [
                    {"pos": (14, 2), "type": "red"},
                    {"pos": (14, 4), "type": "blue"},
                    {"pos": (14, 6), "type": "green"},
                ],
                "obstacles": [
                    (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8)
                ],
            },
            # Level 3
            {
                "player": (1, 1),
                "blocks": [
                    {"pos": (3, 3), "type": "red"},
                    {"pos": (3, 5), "type": "blue"},
                    {"pos": (5, 4), "type": "green"},
                    {"pos": (10, 2), "type": "yellow"},
                ],
                "goals": [
                    {"pos": (13, 7), "type": "red"},
                    {"pos": (13, 1), "type": "blue"},
                    {"pos": (1, 8), "type": "green"},
                    {"pos": (7, 4), "type": "yellow"},
                ],
                "obstacles": [
                    (i, 0) for i in range(16)
                ] + [
                    (i, 9) for i in range(16)
                ] + [
                    (0, i) for i in range(1, 9)
                ] + [
                    (15, i) for i in range(1, 9)
                ] + [
                    (7, 3), (8, 3), (7, 5), (8, 5)
                ],
            },
        ]

    def _load_level(self, level_index):
        if level_index >= len(self._level_definitions):
            return False  # All levels completed

        level_data = self._level_definitions[level_index]
        self.player_pos = level_data["player"]
        self.obstacles = level_data["obstacles"]
        
        self.blocks = []
        block_goals = {g['type']: g['pos'] for g in level_data['goals']}
        for b in level_data['blocks']:
            block_type = b['type']
            self.blocks.append({
                "pos": b['pos'],
                "type": block_type,
                "color": self.BLOCK_COLORS[block_type],
                "goal": block_goals[block_type],
                "goal_color": self.GOAL_COLORS[block_type],
                "on_goal": False,
            })
        return True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.TOTAL_TIME
        self.current_level = 0
        self._load_level(self.current_level)
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]  # 0-4: none/up/down/left/right
        reward = -0.1  # Cost per step
        self.steps += 1
        self.time_left = max(0, self.time_left - self.TIME_PER_STEP)

        # --- Handle Player Movement and Pushing ---
        if movement > 0:
            moves = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
            dx, dy = moves[movement]
            
            # Store pre-move distances for reward calculation
            old_distances = {i: self._manhattan_dist(b['pos'], b['goal']) for i, b in enumerate(self.blocks)}
            moved_blocks_indices = self._handle_push(dx, dy)
            
            # Calculate distance-based reward for moved blocks
            for i in moved_blocks_indices:
                block = self.blocks[i]
                new_dist = self._manhattan_dist(block['pos'], block['goal'])
                # Reward is +1 for each grid cell closer, -1 for each cell further
                reward += old_distances[i] - new_dist

        # --- Check for Blocks on Goals ---
        level_complete = True
        for block in self.blocks:
            was_on_goal = block["on_goal"]
            is_on_goal = block["pos"] == block["goal"]
            block["on_goal"] = is_on_goal
            if not was_on_goal and is_on_goal:
                reward += 10  # Event-based reward for placing a block
                # sound effect: block placed
            if not is_on_goal:
                level_complete = False

        # --- Check for Level/Game Completion ---
        if level_complete:
            reward += 100  # Level complete reward
            # sound effect: level complete
            self.current_level += 1
            if not self._load_level(self.current_level):
                self.game_over = True  # All levels finished, game won
            # Reset block on_goal status for new level
            for block in self.blocks:
                block['on_goal'] = block['pos'] == block['goal']

        # --- Check Termination Conditions ---
        terminated = self.game_over
        if self.time_left <= 0:
            reward -= 50  # Time out penalty
            terminated = True
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_push(self, dx, dy):
        player_x, player_y = self.player_pos
        next_pos = (player_x + dx, player_y + dy)
        
        # Check boundaries
        if not (0 <= next_pos[0] < self.GRID_SIZE[0] and 0 <= next_pos[1] < self.GRID_SIZE[1]):
            return []
        
        # Check for obstacles
        if next_pos in self.obstacles:
            return []

        # Check for blocks and chain reactions
        block_at_next = self._get_block_at(next_pos)
        if not block_at_next:
            self.player_pos = next_pos
            # sound effect: player move
            return []

        # Push attempt
        push_chain = [block_at_next]
        current_pos = next_pos
        
        while True:
            check_pos = (current_pos[0] + dx, current_pos[1] + dy)
            
            # Check for wall collision
            if not (0 <= check_pos[0] < self.GRID_SIZE[0] and 0 <= check_pos[1] < self.GRID_SIZE[1]):
                return [] # Push blocked by wall
            
            # Check for obstacle collision
            if check_pos in self.obstacles:
                return [] # Push blocked by obstacle

            # Check for another block
            next_block_in_chain = self._get_block_at(check_pos)
            if next_block_in_chain:
                push_chain.append(next_block_in_chain)
                current_pos = check_pos
            else:
                # End of chain, push is successful
                moved_indices = []
                for block in reversed(push_chain):
                    block['pos'] = (block['pos'][0] + dx, block['pos'][1] + dy)
                    moved_indices.append(self._get_block_index(block))
                self.player_pos = next_pos
                # sound effect: block push
                return moved_indices
    
    def _get_block_at(self, pos):
        for block in self.blocks:
            if block['pos'] == pos:
                return block
        return None

    def _get_block_index(self, block_to_find):
        for i, block in enumerate(self.blocks):
            if block is block_to_find:
                return i
        return -1

    def _manhattan_dist(self, pos1, pos2):
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
            "time_left": self.time_left,
            "level": self.current_level + 1,
            "blocks_on_goals": sum(1 for b in self.blocks if b['on_goal']),
            "total_blocks": len(self.blocks),
        }
        
    def _render_game(self):
        grid_offset_x = (self.WIDTH - self.GRID_SIZE[0] * self.CELL_SIZE) // 2
        grid_offset_y = (self.HEIGHT - self.GRID_SIZE[1] * self.CELL_SIZE) // 2

        # Draw grid lines
        for x in range(self.GRID_SIZE[0] + 1):
            px = grid_offset_x + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, grid_offset_y), (px, grid_offset_y + self.GRID_SIZE[1] * self.CELL_SIZE))
        for y in range(self.GRID_SIZE[1] + 1):
            py = grid_offset_y + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (grid_offset_x, py), (grid_offset_x + self.GRID_SIZE[0] * self.CELL_SIZE, py))

        # Helper to convert grid to pixel coords
        def to_pixels(pos):
            return (
                grid_offset_x + pos[0] * self.CELL_SIZE,
                grid_offset_y + pos[1] * self.CELL_SIZE,
            )
        
        # Draw goals
        for block in self.blocks:
            px, py = to_pixels(block['goal'])
            pygame.draw.rect(self.screen, block['goal_color'], (px, py, self.CELL_SIZE, self.CELL_SIZE))

        # Draw obstacles
        for obs_pos in self.obstacles:
            px, py = to_pixels(obs_pos)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, (px+4, py+4, self.CELL_SIZE-8, self.CELL_SIZE-8))

        # Draw blocks
        for block in self.blocks:
            px, py = to_pixels(block['pos'])
            padding = 2 if block['on_goal'] else 4
            pygame.draw.rect(self.screen, block['color'], (px + padding, py + padding, self.CELL_SIZE - padding * 2, self.CELL_SIZE - padding * 2), border_radius=4)
            # Add a subtle highlight
            highlight_color = tuple(min(255, c+40) for c in block['color'])
            pygame.draw.rect(self.screen, highlight_color, (px + padding + 2, py + padding + 2, self.CELL_SIZE - padding*2 - 8, 4), border_radius=2)


        # Draw player
        px, py = to_pixels(self.player_pos)
        padding = 8
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (px + padding, py + padding, self.CELL_SIZE - padding * 2, self.CELL_SIZE - padding * 2), border_radius=3)

    def _render_ui(self):
        # Time
        time_text = f"TIME: {int(self.time_left)}"
        time_surf = self.font_small.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (self.WIDTH - time_surf.get_width() - 10, 10))

        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Level
        level_text = f"LEVEL {self.current_level + 1}"
        level_surf = self.font_small.render(level_text, True, self.COLOR_TEXT)
        self.screen.blit(level_surf, (self.WIDTH // 2 - level_surf.get_width() // 2, self.HEIGHT - 30))

        # Game Over / Win message
        if self.game_over:
            all_complete = self.current_level >= len(self._level_definitions)
            message = "YOU WIN!" if all_complete else "GAME OVER"
            color = (100, 255, 100) if all_complete else (255, 100, 100)
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg_surf = self.font_large.render(message, True, color)
            self.screen.blit(msg_surf, (self.WIDTH // 2 - msg_surf.get_width() // 2, self.HEIGHT // 2 - msg_surf.get_height() // 2))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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

# Example of how to run the environment
if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame window setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Pusher Puzzle")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op, no buttons
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
                    action = [0, 0, 0] # Don't process a move on reset frame
                elif event.key == pygame.K_ESCAPE:
                    running = False
        
        # Only step if an action was taken
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")
            if terminated:
                print("Game Over! Press 'r' to restart.")

        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate
        
    env.close()