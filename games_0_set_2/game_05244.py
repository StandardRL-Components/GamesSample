
# Generated: 2025-08-28T04:25:17.031741
# Source Brief: brief_05244.md
# Brief Index: 5244

        
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
        "Controls: Arrow keys to move. Space to hide in designated spots."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated crypt, evading a relentless monster, to find the exit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_WIDTH = 16
        self.GRID_HEIGHT = 10
        self.CELL_SIZE = 40
        self.MAX_STEPS = 1000
        self.INITIAL_HEALTH = 5
        self.HIDING_SPOTS_PER_ROOM = 2

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_WALL = (60, 60, 70)
        self.COLOR_FLOOR = (40, 40, 50)
        self.COLOR_HIDING_SPOT = (30, 30, 40)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (200, 200, 255, 50)
        self.COLOR_MONSTER = (255, 50, 50)
        self.COLOR_MONSTER_GLOW = (255, 50, 50, 70)
        self.COLOR_EXIT = (255, 220, 0)
        self.COLOR_HEALTH_FG = (50, 205, 50)
        self.COLOR_HEALTH_BG = (139, 0, 0)
        self.COLOR_TEXT = (220, 220, 220)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Initialize state variables
        self.grid = None
        self.player_pos = None
        self.monster_pos = None
        self.exit_pos = None
        self.hiding_spots = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_health = 0
        self.current_room = 0
        self.hiding_uses_left = 0
        self.is_hiding = False
        self.last_space_press = False
        self.np_random = None

        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_health = self.INITIAL_HEALTH
        self.current_room = 1
        self.is_hiding = False
        self.last_space_press = False

        self._generate_room()
        
        return self._get_observation(), self._get_info()

    def _generate_room(self):
        # 1: Wall, 0: Floor
        self.grid = np.ones((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)

        # Carve a path
        start_y = self.np_random.integers(1, self.GRID_HEIGHT - 1)
        self.player_pos = np.array([1, start_y])
        self.grid[self.player_pos[0], self.player_pos[1]] = 0

        end_y = self.np_random.integers(1, self.GRID_HEIGHT - 1)
        self.exit_pos = np.array([self.GRID_WIDTH - 2, end_y])

        # Simple random walk to guarantee a path
        path_pos = self.player_pos.copy()
        while not np.array_equal(path_pos, self.exit_pos):
            move_x = np.sign(self.exit_pos[0] - path_pos[0])
            move_y = np.sign(self.exit_pos[1] - path_pos[1])

            if move_x != 0 and (self.np_random.random() > 0.4 or move_y == 0):
                path_pos[0] += move_x
            elif move_y != 0:
                path_pos[1] += move_y
            
            path_pos[0] = np.clip(path_pos[0], 1, self.GRID_WIDTH - 2)
            path_pos[1] = np.clip(path_pos[1], 1, self.GRID_HEIGHT - 2)
            self.grid[path_pos[0], path_pos[1]] = 0

        # Carve more random areas to create a more complex level
        for _ in range(30):
            walk_len = self.np_random.integers(5, 15)
            rx, ry = self.np_random.integers(1, self.GRID_WIDTH-1), self.np_random.integers(1, self.GRID_HEIGHT-1)
            if self.grid[rx, ry] == 0: # Start from an existing floor tile
                for _ in range(walk_len):
                    dx, dy = self.np_random.choice([-1, 0, 1], 2, p=[0.25, 0.5, 0.25])
                    nx, ny = np.clip(rx+dx, 1, self.GRID_WIDTH-2), np.clip(ry+dy, 1, self.GRID_HEIGHT-2)
                    self.grid[nx, ny] = 0
                    rx, ry = nx, ny

        # Place monster far from player
        while True:
            mx, my = self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT)
            if self.grid[mx, my] == 0 and np.linalg.norm(np.array([mx, my]) - self.player_pos) > 5:
                self.monster_pos = np.array([mx, my])
                break

        # Place hiding spots
        self.hiding_spots = []
        self.hiding_uses_left = self.HIDING_SPOTS_PER_ROOM
        attempts = 0
        floor_tiles = np.argwhere(self.grid == 0)
        self.np_random.shuffle(floor_tiles)
        for hx, hy in floor_tiles:
            h_pos = np.array([hx, hy])
            if not np.array_equal(h_pos, self.player_pos) and not np.array_equal(h_pos, self.exit_pos):
                self.hiding_spots.append(h_pos)
                if len(self.hiding_spots) >= self.HIDING_SPOTS_PER_ROOM:
                    break
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        # --- PLAYER TURN ---
        
        # Check for hide action (on rising edge of space press)
        hide_action = space_held and not self.last_space_press
        self.last_space_press = space_held
        
        can_hide = any(np.array_equal(self.player_pos, spot) for spot in self.hiding_spots)

        if hide_action and can_hide and self.hiding_uses_left > 0:
            self.is_hiding = True
            self.hiding_uses_left -= 1
            reward -= 2.0  # Cost for using a hiding spot
            reward -= 0.5  # Cost for the step spent hiding
            # SFX: whoosh_hide
        else:
            # Handle movement
            old_pos = self.player_pos.copy()
            old_dist_to_exit = np.linalg.norm(old_pos - self.exit_pos)
            
            new_pos = old_pos.copy()
            if movement == 1: new_pos[1] -= 1 # Up
            elif movement == 2: new_pos[1] += 1 # Down
            elif movement == 3: new_pos[0] -= 1 # Left
            elif movement == 4: new_pos[0] += 1 # Right

            # Check boundaries and walls
            if movement != 0 and 0 <= new_pos[0] < self.GRID_WIDTH and 0 <= new_pos[1] < self.GRID_HEIGHT and self.grid[new_pos[0], new_pos[1]] == 0:
                self.player_pos = new_pos
                new_dist_to_exit = np.linalg.norm(self.player_pos - self.exit_pos)
                if new_dist_to_exit < old_dist_to_exit:
                    reward += 0.1
                else:
                    reward -= 0.2
        
        # --- MONSTER TURN ---
        if self.is_hiding:
            # SFX: monster_confused
            self.is_hiding = False # Hiding lasts one turn
        else:
            # Monster moves towards player using simple greedy pathfinding
            def move_monster():
                best_move = self.monster_pos
                min_dist = np.linalg.norm(self.monster_pos - self.player_pos)
                for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                    next_pos = self.monster_pos + np.array([dx, dy])
                    if 0 <= next_pos[0] < self.GRID_WIDTH and 0 <= next_pos[1] < self.GRID_HEIGHT and self.grid[next_pos[0], next_pos[1]] == 0:
                        dist = np.linalg.norm(next_pos - self.player_pos)
                        if dist < min_dist:
                            min_dist = dist
                            best_move = next_pos
                self.monster_pos = best_move
            
            move_monster()
            # SFX: monster_step

            # Difficulty scaling: chance for extra move
            monster_extra_move_chance = ((self.current_room - 1) // 2) * 0.05
            if self.np_random.random() < monster_extra_move_chance:
                move_monster()

        # --- CHECK CONSEQUENCES ---
        terminated = False
        
        if np.array_equal(self.player_pos, self.monster_pos):
            self.player_health -= 1
            # SFX: player_hurt, monster_growl
            if self.player_health <= 0:
                reward -= 100
                terminated = True
                self.game_over = True
                # SFX: game_over_lose
        
        if not terminated and np.array_equal(self.player_pos, self.exit_pos):
            reward += 100
            self.score += 100
            self.current_room += 1
            self._generate_room()
            # SFX: level_up
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                color = self.COLOR_WALL if self.grid[x, y] == 1 else self.COLOR_FLOOR
                pygame.draw.rect(self.screen, color, rect)

        # Draw hiding spots
        for spot in self.hiding_spots:
            rect = pygame.Rect(spot[0] * self.CELL_SIZE, spot[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_HIDING_SPOT, rect)

        # Draw exit
        exit_rect = pygame.Rect(self.exit_pos[0] * self.CELL_SIZE, self.exit_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)
        
        # Draw monster
        monster_center_x = int(self.monster_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2)
        monster_center_y = int(self.monster_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2)
        glow_radius = int(self.CELL_SIZE * (0.8 + 0.1 * math.sin(self.steps * 0.2)))
        pygame.gfxdraw.filled_circle(self.screen, monster_center_x, monster_center_y, glow_radius, self.COLOR_MONSTER_GLOW)
        monster_rect = pygame.Rect(self.monster_pos[0] * self.CELL_SIZE + 5, self.monster_pos[1] * self.CELL_SIZE + 5, self.CELL_SIZE - 10, self.CELL_SIZE - 10)
        pygame.draw.rect(self.screen, self.COLOR_MONSTER, monster_rect)
        
        # Draw player
        player_center_x = int(self.player_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2)
        player_center_y = int(self.player_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2)
        pygame.gfxdraw.filled_circle(self.screen, player_center_x, player_center_y, int(self.CELL_SIZE*0.6), self.COLOR_PLAYER_GLOW)
        player_rect = pygame.Rect(self.player_pos[0] * self.CELL_SIZE + 8, self.player_pos[1] * self.CELL_SIZE + 8, self.CELL_SIZE - 16, self.CELL_SIZE - 16)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        if self.is_hiding:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, 3)

    def _render_ui(self):
        # Health bar
        health_ratio = max(0, self.player_health / self.INITIAL_HEALTH)
        health_bar_width = 150
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (10, 10, health_bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (10, 10, int(health_bar_width * health_ratio), 20))

        # Room number
        room_text = self.font_small.render(f"Room: {self.current_room}", True, self.COLOR_TEXT)
        self.screen.blit(room_text, (self.SCREEN_WIDTH - room_text.get_width() - 10, 10))

        # Hiding spots uses
        hide_text = self.font_small.render(f"Hiding Spots: {self.hiding_uses_left}", True, self.COLOR_TEXT)
        self.screen.blit(hide_text, (self.SCREEN_WIDTH / 2 - hide_text.get_width() / 2, self.SCREEN_HEIGHT - 30))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            status_text_str = "GAME OVER"
            if self.steps >= self.MAX_STEPS:
                status_text_str = "TIME UP"
            
            status_text = self.font_large.render(status_text_str, True, self.COLOR_MONSTER)
            text_rect = status_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(status_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "current_room": self.current_room,
            "player_pos": self.player_pos.tolist(),
            "monster_pos": self.monster_pos.tolist(),
        }

    def close(self):
        pygame.font.quit()
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
    # This block allows you to run the file directly to test the environment
    # It demonstrates a random agent playing the game
    # For human play, you would need to add a "human" render mode and capture keyboard events.
    
    print("Initializing environment...")
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    print("Running a short simulation with a random agent...")
    total_reward = 0
    for i in range(1, 501):
        action = env.action_space.sample() # Random agent
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if i % 50 == 0:
            print(f"Step {i}: Action={action}, Reward={reward:.2f}, Total Reward={total_reward:.2f}, "
                  f"Score={info['score']:.1f}, Health={info['player_health']}, Room={info['current_room']}")
            
        if terminated or truncated:
            print(f"\nEpisode finished after {i} steps.")
            print(f"Final Info: {info}")
            break
    
    env.close()
    print("\nSimulation finished.")