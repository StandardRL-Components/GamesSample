
# Generated: 2025-08-27T17:53:49.313329
# Source Brief: brief_01672.md
# Brief Index: 1672

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import random
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to navigate the maze. Collect numbers in ascending order (1-10)."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist puzzle game. Find the shortest path to collect all numbers in order before you run out of moves."
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

        # Game Constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.MAZE_WIDTH = 16  # 640 / 40
        self.MAZE_HEIGHT = 10  # 400 / 40
        self.CELL_SIZE = 40
        self.MAX_MOVES = 20
        self.NUM_TARGETS = 10

        # Colors
        self.COLOR_BG = (220, 220, 220)
        self.COLOR_WALL = (60, 60, 60)
        self.COLOR_PLAYER = (0, 122, 255)
        self.COLOR_NUMBER = (255, 255, 255)
        self.COLOR_NUMBER_OUTLINE = (0, 0, 0)
        self.COLOR_UI_TEXT = (20, 20, 20)
        self.COLOR_CHECKMARK = (46, 204, 113)
        self.COLOR_UI_BOX = (200, 200, 200)

        # Fonts
        self.ui_font = pygame.font.Font(None, 28)
        self.number_font = pygame.font.Font(None, 32)
        
        # State variables (initialized in reset)
        self.maze = None
        self.player_pos = None
        self.numbers = None
        self.next_number_to_collect = None
        self.moves_left = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.np_random = None

        # Initialize state
        self.reset()
        
        # Run self-check
        self.validate_implementation()

    def _generate_maze(self, width, height):
        # Maze grid: each cell has walls on all sides initially
        maze = [[{'N': True, 'S': True, 'E': True, 'W': True, 'visited': False} for _ in range(width)] for _ in range(height)]
        
        # Use a stack for DFS
        stack = []
        # Start at a random cell
        start_x, start_y = self.np_random.integers(0, width), self.np_random.integers(0, height)
        maze[start_y][start_x]['visited'] = True
        stack.append((start_x, start_y))

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            # Check North
            if cy > 0 and not maze[cy - 1][cx]['visited']:
                neighbors.append('N')
            # Check South
            if cy < height - 1 and not maze[cy + 1][cx]['visited']:
                neighbors.append('S')
            # Check East
            if cx < width - 1 and not maze[cy][cx + 1]['visited']:
                neighbors.append('E')
            # Check West
            if cx > 0 and not maze[cy][cx - 1]['visited']:
                neighbors.append('W')

            if neighbors:
                # Choose a random unvisited neighbor
                direction = self.np_random.choice(neighbors)
                if direction == 'N':
                    nx, ny = cx, cy - 1
                    maze[cy][cx]['N'] = False
                    maze[ny][nx]['S'] = False
                elif direction == 'S':
                    nx, ny = cx, cy + 1
                    maze[cy][cx]['S'] = False
                    maze[ny][nx]['N'] = False
                elif direction == 'E':
                    nx, ny = cx + 1, cy
                    maze[cy][cx]['E'] = False
                    maze[ny][nx]['W'] = False
                else:  # 'W'
                    nx, ny = cx - 1, cy
                    maze[cy][cx]['W'] = False
                    maze[ny][nx]['E'] = False
                
                maze[ny][nx]['visited'] = True
                stack.append((nx, ny))
            else:
                # Backtrack
                stack.pop()
        return maze

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.maze = self._generate_maze(self.MAZE_WIDTH, self.MAZE_HEIGHT)
        
        floor_cells = [(x, y) for y in range(self.MAZE_HEIGHT) for x in range(self.MAZE_WIDTH)]
        
        placements_indices = self.np_random.choice(
            len(floor_cells),
            size=self.NUM_TARGETS + 1,
            replace=False
        )
        
        player_idx = placements_indices[0]
        self.player_pos = list(floor_cells[player_idx])

        self.numbers = []
        for i in range(self.NUM_TARGETS):
            num_idx = placements_indices[i + 1]
            pos = floor_cells[num_idx]
            self.numbers.append({"value": i + 1, "pos": pos})
            
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES
        self.next_number_to_collect = 1
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        terminated = False
        
        old_pos = tuple(self.player_pos)
        
        # Only process movement if a direction is chosen
        if movement > 0:
            self.moves_left -= 1
            px, py = self.player_pos
            
            # Check for wall collisions before moving
            can_move = False
            if movement == 1 and not self.maze[py][px]['N']: # Up
                self.player_pos[1] -= 1
                can_move = True
            elif movement == 2 and not self.maze[py][px]['S']: # Down
                self.player_pos[1] += 1
                can_move = True
            elif movement == 3 and not self.maze[py][px]['W']: # Left
                self.player_pos[0] -= 1
                can_move = True
            elif movement == 4 and not self.maze[py][px]['E']: # Right
                self.player_pos[0] += 1
                can_move = True
            
            # Distance-based reward
            target_num = self._find_number(self.next_number_to_collect)
            if target_num and can_move:
                target_pos = target_num['pos']
                old_dist = abs(old_pos[0] - target_pos[0]) + abs(old_pos[1] - target_pos[1])
                new_dist = abs(self.player_pos[0] - target_pos[0]) + abs(self.player_pos[1] - target_pos[1])
                if new_dist < old_dist:
                    reward += 1
                elif new_dist > old_dist:
                    reward -= 1
        
        # Check for number collection
        collected_num_idx = -1
        for i, num in enumerate(self.numbers):
            if tuple(self.player_pos) == num['pos']:
                if num['value'] == self.next_number_to_collect:
                    reward += 10
                    self.score += 10
                    self.next_number_to_collect += 1
                    collected_num_idx = i
                    # sfx: correct_collection.wav
                else:
                    reward -= 5
                    self.score -= 5
                    # sfx: wrong_collection.wav
                break
        
        if collected_num_idx != -1:
            self.numbers.pop(collected_num_idx)

        # Check termination conditions
        if self.next_number_to_collect > self.NUM_TARGETS:
            terminated = True
            self.game_over = True
            reward += 50
            self.score += 50
            # sfx: win_game.wav
        elif self.moves_left <= 0:
            terminated = True
            self.game_over = True
            reward -= 20
            self.score -= 20
            # sfx: lose_game.wav

        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _find_number(self, value):
        for num in self.numbers:
            if num['value'] == value:
                return num
        return None

    def _render_text_with_outline(self, surface, text, font, pos, color, outline_color):
        x, y = pos
        text_surface = font.render(text, True, outline_color)
        surface.blit(text_surface, (x - 1, y - 1))
        surface.blit(text_surface, (x + 1, y - 1))
        surface.blit(text_surface, (x - 1, y + 1))
        surface.blit(text_surface, (x + 1, y + 1))
        
        text_surface = font.render(text, True, color)
        surface.blit(text_surface, pos)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        # Render Maze Walls
        for y in range(self.MAZE_HEIGHT):
            for x in range(self.MAZE_WIDTH):
                cell = self.maze[y][x]
                px, py = x * self.CELL_SIZE, y * self.CELL_SIZE
                if cell['N']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py), (px + self.CELL_SIZE, py), 2)
                if cell['S']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py + self.CELL_SIZE), (px + self.CELL_SIZE, py + self.CELL_SIZE), 2)
                if cell['W']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py), (px, py + self.CELL_SIZE), 2)
                if cell['E']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px + self.CELL_SIZE, py), (px + self.CELL_SIZE, py + self.CELL_SIZE), 2)

        # Render Numbers
        for num in self.numbers:
            val_str = str(num['value'])
            text_surf = self.number_font.render(val_str, True, self.COLOR_NUMBER)
            text_rect = text_surf.get_rect(center=(
                int(num['pos'][0] * self.CELL_SIZE + self.CELL_SIZE / 2),
                int(num['pos'][1] * self.CELL_SIZE + self.CELL_SIZE / 2)
            ))
            self._render_text_with_outline(self.screen, val_str, self.number_font, text_rect.topleft, self.COLOR_NUMBER, self.COLOR_NUMBER_OUTLINE)
            
        # Render Player
        player_rect = pygame.Rect(
            int(self.player_pos[0] * self.CELL_SIZE + self.CELL_SIZE * 0.15),
            int(self.player_pos[1] * self.CELL_SIZE + self.CELL_SIZE * 0.15),
            int(self.CELL_SIZE * 0.7),
            int(self.CELL_SIZE * 0.7)
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)

        # Render UI
        # Moves Left
        moves_text = f"Moves: {self.moves_left}"
        moves_surf = self.ui_font.render(moves_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_surf, (10, 10))

        # Score
        score_text = f"Score: {self.score}"
        score_surf = self.ui_font.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (self.SCREEN_WIDTH - score_surf.get_width() - 10, 10))

        # Collected Numbers Tracker
        tracker_width = self.NUM_TARGETS * 25
        start_x = (self.SCREEN_WIDTH - tracker_width) // 2
        for i in range(1, self.NUM_TARGETS + 1):
            num_text = str(i)
            is_collected = i < self.next_number_to_collect
            color = self.COLOR_CHECKMARK if is_collected else self.COLOR_UI_TEXT
            
            text_surf = self.ui_font.render(num_text, True, color)
            text_pos_x = start_x + (i - 1) * 25
            self.screen.blit(text_surf, (text_pos_x, 10))
            if is_collected:
                # Draw checkmark
                p1 = (text_pos_x + 15, 12)
                p2 = (text_pos_x + 18, 22)
                p3 = (text_pos_x + 23, 8)
                pygame.draw.lines(self.screen, self.COLOR_CHECKMARK, False, [p1, p2, p3], 2)

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "next_number": self.next_number_to_collect,
        }

    def close(self):
        pygame.quit()

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

if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Number Maze")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    
    print("\n" + "="*30)
    print("      MANUAL PLAY MODE")
    print("="*30)
    print(env.user_guide)
    print("Press ESC or close the window to quit.\n")

    while not terminated:
        # Convert observation for display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Get user input
        action = [0, 0, 0] # Default to no-op
        
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    terminated = True
                elif event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset game
                    print("Resetting game...")
                    obs, info = env.reset()
                    continue

                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")

                if terminated:
                    print("\nGAME OVER!")
                    if info['score'] > 0:
                        print(f"Final Score: {info['score']} - You won!")
                    else:
                        print(f"Final Score: {info['score']} - You ran out of moves!")
                    
                    # Show final frame
                    frame = np.transpose(obs, (1, 0, 2))
                    surf = pygame.surfarray.make_surface(frame)
                    screen.blit(surf, (0, 0))
                    pygame.display.flip()
                    pygame.time.wait(3000) # Wait 3 seconds before closing
    
    env.close()