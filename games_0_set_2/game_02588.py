
# Generated: 2025-08-27T20:49:25.396647
# Source Brief: brief_02588.md
# Brief Index: 2588

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move your character (white square) one tile at a time. "
        "Space and Shift are not used."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a dark crypt, collect all 3 yellow keys, and reach the green exit before the timer runs out. "
        "Avoid the red patrolling spirits!"
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
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # Game Constants
        self.TILE_SIZE = 20
        self.GRID_WIDTH = self.screen_width // self.TILE_SIZE
        self.GRID_HEIGHT = self.screen_height // self.TILE_SIZE
        self.MAX_STEPS = 60
        self.NUM_ENEMIES = 5
        self.NUM_KEYS = 3
        self.NUM_WALLS = 45

        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_WALL = (50, 60, 70)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_FLICKER = (180, 20, 20)
        self.COLOR_KEY = (255, 220, 0)
        self.COLOR_EXIT_CLOSED = (0, 100, 50)
        self.COLOR_EXIT_OPEN = (0, 255, 120)
        self.COLOR_TEXT = (200, 200, 200)

        # Font
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 22, bold=True)
            self.font_key = pygame.font.SysFont("Consolas", 18, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 28)
            self.font_key = pygame.font.SysFont(None, 24)

        # Game state variables (initialized in reset)
        self.player_pos = None
        self.enemy_states = None
        self.keys = None
        self.exit_pos = None
        self.walls = None
        
        self.steps = 0
        self.score = 0.0
        self.timer = 0
        self.keys_collected = 0
        self.exit_open = False
        self.game_over = False
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0.0
        self.timer = self.MAX_STEPS
        self.keys_collected = 0
        self.exit_open = False
        self.game_over = False

        self._generate_level()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        # This loop ensures a valid level is generated
        while True:
            all_coords = set((x, y) for x in range(1, self.GRID_WIDTH - 1) for y in range(1, self.GRID_HEIGHT - 1))

            self.walls = set()
            for x in range(self.GRID_WIDTH):
                self.walls.add((x, 0))
                self.walls.add((x, self.GRID_HEIGHT - 1))
            for y in range(self.GRID_HEIGHT):
                self.walls.add((0, y))
                self.walls.add((self.GRID_WIDTH - 1, y))

            floor_tiles = list(all_coords)
            wall_indices = self.np_random.choice(len(floor_tiles), size=self.NUM_WALLS, replace=False)
            for i in wall_indices:
                self.walls.add(floor_tiles[i])

            open_tiles = list(all_coords - self.walls)
            if not open_tiles: continue

            player_idx = self.np_random.integers(len(open_tiles))
            self.player_pos = open_tiles[player_idx]

            q = [self.player_pos]
            reachable = {self.player_pos}
            head = 0
            while head < len(q):
                x, y = q[head]
                head += 1
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if (nx, ny) not in self.walls and (nx, ny) not in reachable:
                        reachable.add((nx, ny))
                        q.append((nx, ny))
            
            spawn_points = list(reachable - {self.player_pos})
            
            num_objects_to_place = self.NUM_KEYS + 1 + self.NUM_ENEMIES
            if len(spawn_points) >= num_objects_to_place:
                obj_indices = self.np_random.choice(len(spawn_points), size=num_objects_to_place, replace=False)
                
                self.keys = {spawn_points[i] for i in obj_indices[:self.NUM_KEYS]}
                self.exit_pos = spawn_points[obj_indices[self.NUM_KEYS]]
                
                self.enemy_states = []
                for i in range(self.NUM_ENEMIES):
                    start_pos = spawn_points[obj_indices[self.NUM_KEYS + 1 + i]]
                    patrol_size = self.np_random.integers(2, 5)
                    path = []
                    for _ in range(patrol_size): path.append((1, 0))
                    for _ in range(patrol_size): path.append((0, 1))
                    for _ in range(patrol_size): path.append((-1, 0))
                    for _ in range(patrol_size): path.append((0, -1))
                    
                    self.enemy_states.append({
                        'pos': start_pos,
                        'path': path,
                        'index': self.np_random.integers(len(path))
                    })
                break # Valid level generated, exit while loop
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        reward = -0.1  # Time penalty
        terminated = False

        # 1. Unpack and process player action
        movement = action[0]
        px, py = self.player_pos
        if movement == 1: py -= 1  # Up
        elif movement == 2: py += 1  # Down
        elif movement == 3: px -= 1  # Left
        elif movement == 4: px += 1  # Right
        
        if (px, py) not in self.walls:
            self.player_pos = (px, py)

        # 2. Update enemy positions
        for enemy in self.enemy_states:
            ex, ey = enemy['pos']
            dx, dy = enemy['path'][enemy['index']]
            nex, ney = ex + dx, ey + dy
            if (nex, ney) not in self.walls:
                enemy['pos'] = (nex, ney)
            enemy['index'] = (enemy['index'] + 1) % len(enemy['path'])
        
        # 3. Check for game events and rewards
        if self.player_pos in self.keys:
            self.keys.remove(self.player_pos)
            self.keys_collected += 1
            reward += 1.0
            # sfx: key_pickup.wav
            if self.keys_collected == self.NUM_KEYS:
                self.exit_open = True
                # sfx: exit_unlocked.wav

        # 4. Check for termination conditions
        # Enemy collision
        for enemy in self.enemy_states:
            if self.player_pos == enemy['pos']:
                reward = -100.0
                terminated = True
                self.game_over = True
                # sfx: player_death.wav
                break
        
        if not terminated:
            # Win condition
            if self.exit_open and self.player_pos == self.exit_pos:
                reward = 100.0
                terminated = True
                self.game_over = True
                # sfx: level_complete.wav
            else:
                # Time out
                self.timer -= 1
                if self.timer <= 0:
                    reward = -50.0
                    terminated = True
                    self.game_over = True
                    # sfx: timeout_alarm.wav
        
        self.steps += 1
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw walls
        for x, y in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, (x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE))

        # Draw exit
        exit_color = self.COLOR_EXIT_OPEN if self.exit_open else self.COLOR_EXIT_CLOSED
        ex, ey = self.exit_pos
        pygame.gfxdraw.box(self.screen, (ex * self.TILE_SIZE, ey * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE), exit_color)

        # Draw keys
        for x, y in self.keys:
            pygame.gfxdraw.box(self.screen, (x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE), self.COLOR_KEY)

        # Draw enemies with flicker
        flicker = (self.steps // 4) % 2 == 0
        enemy_color = self.COLOR_ENEMY if flicker else self.COLOR_ENEMY_FLICKER
        for enemy in self.enemy_states:
            ex, ey = enemy['pos']
            pygame.gfxdraw.box(self.screen, (ex * self.TILE_SIZE, ey * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE), enemy_color)
        
        # Draw player
        px, py = self.player_pos
        player_rect = (px * self.TILE_SIZE, py * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pygame.gfxdraw.box(self.screen, player_rect, self.COLOR_PLAYER)

    def _render_ui(self):
        # UI Background panels
        pygame.draw.rect(self.screen, (0,0,0,150), (0, 0, 140, 40))
        pygame.draw.rect(self.screen, (0,0,0,150), (self.screen_width - 120, 0, 120, 40))

        # Key Icon and Count
        key_icon_rect = (10, 10, self.TILE_SIZE, self.TILE_SIZE)
        pygame.gfxdraw.box(self.screen, key_icon_rect, self.COLOR_KEY)
        key_text = self.font_ui.render(f"{self.keys_collected}/{self.NUM_KEYS}", True, self.COLOR_TEXT)
        self.screen.blit(key_text, (40, 8))

        # Timer
        timer_text = self.font_ui.render(f"TIME: {self.timer}", True, self.COLOR_TEXT)
        text_rect = timer_text.get_rect(topright=(self.screen_width - 10, 8))
        self.screen.blit(timer_text, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "keys_collected": self.keys_collected,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
    # This block allows you to play the game manually
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Prevents Pygame from opening a window if not needed
    
    # To play manually, comment out the line above and uncomment the block below.
    # You will need to install pygame: pip install pygame
    
    # --- Manual Play Block ---
    # env = GameEnv(render_mode="rgb_array")
    # obs, info = env.reset()
    # screen = pygame.display.set_mode((640, 400))
    # pygame.display.set_caption("Crypt Escape")
    # running = True
    # while running:
    #     action = np.array([0, 0, 0]) # Default action: no-op
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False
    #         if event.type == pygame.KEYDOWN:
    #             if event.key == pygame.K_UP:
    #                 action[0] = 1
    #             elif event.key == pygame.K_DOWN:
    #                 action[0] = 2
    #             elif event.key == pygame.K_LEFT:
    #                 action[0] = 3
    #             elif event.key == pygame.K_RIGHT:
    #                 action[0] = 4
    #             elif event.key == pygame.K_ESCAPE:
    #                 running = False
            
    #             obs, reward, terminated, truncated, info = env.step(action)
    #             print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")
                
    #             if terminated:
    #                 print("Game Over! Resetting...")
    #                 obs, info = env.reset()
    
    #     # Draw the observation to the screen
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
    
    # env.close()
    # --- End Manual Play Block ---

    # Standard test to ensure the environment runs
    env = GameEnv()
    obs, info = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            print(f"Episode finished after {_ + 1} steps. Final score: {info['score']:.2f}")
            obs, info = env.reset()
    env.close()