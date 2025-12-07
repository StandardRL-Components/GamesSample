
# Generated: 2025-08-28T05:23:41.795540
# Source Brief: brief_05564.md
# Brief Index: 5564

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Arrow keys to move. Spacebar to attack in your last moved direction."
    )

    # Short, user-facing description of the game
    game_description = (
        "Explore a procedurally generated dungeon, battling enemies and collecting gold to reach the exit on the 5th floor."
    )

    # Frames advance only on action
    auto_advance = False
    
    # --- Constants ---
    # Colors
    COLOR_BG = (20, 15, 25)
    COLOR_WALL = (60, 50, 70)
    COLOR_FLOOR = (90, 80, 100)
    COLOR_PLAYER = (50, 220, 100)
    COLOR_PLAYER_ACCENT = (150, 255, 180)
    COLOR_ENEMY = (220, 50, 50)
    COLOR_ENEMY_ACCENT = (255, 150, 150)
    COLOR_GOLD = (255, 215, 0)
    COLOR_EXIT = (180, 50, 255)
    COLOR_TEXT = (240, 240, 240)
    COLOR_ATTACK_SLASH = (255, 255, 255)
    COLOR_DAMAGE_FLASH = (255, 255, 255)
    
    # Screen and Grid
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    UI_HEIGHT = 40
    GAME_HEIGHT = SCREEN_HEIGHT - UI_HEIGHT
    TILE_SIZE = 20
    GRID_WIDTH = SCREEN_WIDTH // TILE_SIZE
    GRID_HEIGHT = GAME_HEIGHT // TILE_SIZE

    # Game parameters
    MAX_STEPS = 1000
    TOTAL_FLOORS = 5
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_main = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player = {}
        self.enemies = []
        self.gold = []
        self.grid = np.array([])
        self.floor_number = 1
        self.exit_pos = (0, 0)
        self.vfx = []
        
        self.reset()
        self.validate_implementation()

    def _generate_floor(self):
        # 1. Start with a grid of walls
        self.grid = np.ones((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        
        # 2. Carve out a path using random walk
        start_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.player['x'], self.player['y'] = start_pos
        self.grid[start_pos] = 0 # 0 for floor

        path = {start_pos}
        walker_pos = start_pos
        path_len = (self.GRID_WIDTH * self.GRID_HEIGHT) // 4
        
        for _ in range(path_len):
            dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            nx, ny = walker_pos[0] + dx, walker_pos[1] + dy
            if 1 <= nx < self.GRID_WIDTH - 1 and 1 <= ny < self.GRID_HEIGHT - 1:
                walker_pos = (nx, ny)
                self.grid[walker_pos] = 0
                path.add(walker_pos)

        self.exit_pos = walker_pos
        
        # 3. Add some rooms
        num_rooms = random.randint(5, 10)
        for _ in range(num_rooms):
            rw, rh = random.randint(3, 6), random.randint(3, 6)
            rx, ry = random.randint(1, self.GRID_WIDTH - rw - 1), random.randint(1, self.GRID_HEIGHT - rh - 1)
            for y in range(ry, ry + rh):
                for x in range(rx, rx + rw):
                    self.grid[x, y] = 0
                    path.add((x, y))

        # 4. Populate with enemies and gold
        self.enemies = []
        self.gold = []
        
        available_tiles = list(path - {start_pos, self.exit_pos})
        random.shuffle(available_tiles)

        num_enemies = 20
        enemy_health = 4 + self.floor_number
        for _ in range(num_enemies):
            if not available_tiles: break
            x, y = available_tiles.pop()
            self.enemies.append({
                'x': x, 'y': y, 'health': enemy_health, 
                'patrol_idx': random.randint(0, 3),
                'flash': 0
            })

        num_gold = random.randint(10, 20)
        for _ in range(num_gold):
            if not available_tiles: break
            x, y = available_tiles.pop()
            self.gold.append({'x': x, 'y': y})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.floor_number = 1
        
        self.player = {
            'x': 0, 'y': 0, 
            'health': 10, 'max_health': 10,
            'facing': (0, 1), # Down initially
            'flash': 0
        }
        
        self._generate_floor()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.vfx = [] # Clear visual effects each step

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Player Turn ---
        player_moved = False
        old_dist_to_exit = abs(self.player['x'] - self.exit_pos[0]) + abs(self.player['y'] - self.exit_pos[1])
        
        # 1. Handle Movement
        if movement != 0:
            move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)} # Up, Down, Left, Right
            dx, dy = move_map[movement]
            self.player['facing'] = (dx, dy)
            
            nx, ny = self.player['x'] + dx, self.player['y'] + dy
            
            if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.grid[nx, ny] == 0:
                is_enemy_at_target = any(e['x'] == nx and e['y'] == ny for e in self.enemies)
                if not is_enemy_at_target:
                    self.player['x'], self.player['y'] = nx, ny
                    player_moved = True
                else: # Bumped into an enemy
                    self.player['health'] -= 1
                    self.player['flash'] = 2 # Flash for 2 frames (current and next)
                    reward -= 0.2
                    # // Sound: Player bump damage

        # 2. Handle Attack
        if space_held:
            ax, ay = self.player['x'] + self.player['facing'][0], self.player['y'] + self.player['facing'][1]
            self.vfx.append({'type': 'slash', 'x': ax, 'y': ay, 'facing': self.player['facing']})
            # // Sound: Player attack miss
            
            for enemy in self.enemies:
                if enemy['x'] == ax and enemy['y'] == ay:
                    enemy['health'] -= 1
                    enemy['flash'] = 2
                    # // Sound: Player attack hit
                    if enemy['health'] <= 0:
                        reward += 1.0
                        self.score += 100
                        # // Sound: Enemy defeat
                    break

        # Post-move checks
        if player_moved:
            # Gold collection
            for g in self.gold:
                if g['x'] == self.player['x'] and g['y'] == self.player['y']:
                    self.gold.remove(g)
                    reward += 0.5
                    self.score += 50
                    # // Sound: Gold collect
                    break
            
            # Reward for moving towards/away from exit
            new_dist_to_exit = abs(self.player['x'] - self.exit_pos[0]) + abs(self.player['y'] - self.exit_pos[1])
            if new_dist_to_exit < old_dist_to_exit:
                reward += 0.1
            else:
                reward -= 0.1

        # --- Enemy Turn ---
        occupied_tiles = {(e['x'], e['y']) for e in self.enemies}
        occupied_tiles.add((self.player['x'], self.player['y']))
        
        patrol_moves = [(0, -1), (1, 0), (0, 1), (-1, 0)] # Up, Right, Down, Left

        self.enemies = [e for e in self.enemies if e['health'] > 0] # Remove defeated enemies

        for enemy in self.enemies:
            # Attack if adjacent
            is_adjacent = abs(enemy['x'] - self.player['x']) + abs(enemy['y'] - self.player['y']) == 1
            if is_adjacent:
                self.player['health'] -= 1
                self.player['flash'] = 2
                # // Sound: Player takes damage
            else: # Patrol
                for _ in range(4): # Try all 4 directions from current patrol index
                    dx, dy = patrol_moves[enemy['patrol_idx']]
                    nx, ny = enemy['x'] + dx, enemy['y'] + dy
                    
                    if (nx, ny) not in occupied_tiles and 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.grid[nx, ny] == 0:
                        occupied_tiles.remove((enemy['x'], enemy['y']))
                        enemy['x'], enemy['y'] = nx, ny
                        occupied_tiles.add((nx, ny))
                        break
                    else:
                        enemy['patrol_idx'] = (enemy['patrol_idx'] + 1) % 4
        
        # --- Update State ---
        self.steps += 1
        
        # Floor transition
        if (self.player['x'], self.player['y']) == self.exit_pos and self.floor_number < self.TOTAL_FLOORS:
            self.floor_number += 1
            self.score += 250
            self._generate_floor()
            # // Sound: Level up
        
        # --- Termination Checks ---
        terminated = False
        if self.player['health'] <= 0:
            reward -= 100
            terminated = True
            self.game_over = True
            # // Sound: Player death
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        if (self.player['x'], self.player['y']) == self.exit_pos and self.floor_number == self.TOTAL_FLOORS:
            reward += 100
            terminated = True
            self.game_over = True
            self.score += 1000
            # // Sound: Victory
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "floor": self.floor_number}

    def _render_game(self):
        ts = self.TILE_SIZE
        
        # 1. Draw grid
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = (x * ts, y * ts + self.UI_HEIGHT, ts, ts)
                color = self.COLOR_FLOOR if self.grid[x, y] == 0 else self.COLOR_WALL
                pygame.draw.rect(self.screen, color, rect)

        # 2. Draw exit
        if self.floor_number == self.TOTAL_FLOORS:
            ex, ey = self.exit_pos
            rect = (ex * ts, ey * ts + self.UI_HEIGHT, ts, ts)
            pygame.draw.rect(self.screen, self.COLOR_EXIT, rect)
            pygame.draw.rect(self.screen, (255,255,255), rect, 1)

        # 3. Draw gold
        for g in self.gold:
            gx, gy = g['x'] * ts + ts // 2, g['y'] * ts + self.UI_HEIGHT + ts // 2
            pygame.draw.circle(self.screen, self.COLOR_GOLD, (gx, gy), ts // 4)

        # 4. Draw enemies
        for e in self.enemies:
            ex, ey = e['x'] * ts, e['y'] * ts + self.UI_HEIGHT
            color = self.COLOR_DAMAGE_FLASH if e['flash'] > 0 else self.COLOR_ENEMY
            pygame.draw.rect(self.screen, color, (ex + 2, ey + 2, ts - 4, ts - 4))
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_ACCENT, (ex + 4, ey + 4, ts - 8, ts - 8))
            if e['flash'] > 0: e['flash'] -= 1
        
        # 5. Draw player
        px, py = self.player['x'] * ts, self.player['y'] * ts + self.UI_HEIGHT
        color = self.COLOR_DAMAGE_FLASH if self.player['flash'] > 0 else self.COLOR_PLAYER
        pygame.draw.rect(self.screen, color, (px + 2, py + 2, ts - 4, ts - 4))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_ACCENT, (px + 4, py + 4, ts - 8, ts - 8))
        if self.player['flash'] > 0: self.player['flash'] -= 1

        # 6. Draw VFX
        for vfx in self.vfx:
            if vfx['type'] == 'slash':
                x, y, facing = vfx['x'], vfx['y'], vfx['facing']
                center_x, center_y = x * ts + ts // 2, y * ts + self.UI_HEIGHT + ts // 2
                
                if abs(facing[0]) == 1: # Horizontal
                    start_pos = (center_x, center_y - ts // 3)
                    end_pos = (center_x, center_y + ts // 3)
                else: # Vertical
                    start_pos = (center_x - ts // 3, center_y)
                    end_pos = (center_x + ts // 3, center_y)
                pygame.draw.line(self.screen, self.COLOR_ATTACK_SLASH, start_pos, end_pos, 3)

    def _render_ui(self):
        # Background bar
        pygame.draw.rect(self.screen, (30, 25, 40), (0, 0, self.SCREEN_WIDTH, self.UI_HEIGHT))
        
        # Health
        health_text = self.font_main.render(f"HP: {self.player['health']}/{self.player['max_health']}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 10))

        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(centerx=self.SCREEN_WIDTH / 2, top=10)
        self.screen.blit(score_text, score_rect)
        
        # Floor
        floor_text = self.font_main.render(f"FLOOR: {self.floor_number}/{self.TOTAL_FLOORS}", True, self.COLOR_TEXT)
        floor_rect = floor_text.get_rect(right=self.SCREEN_WIDTH - 10, top=10)
        self.screen.blit(floor_text, floor_rect)

        # Game Over / Victory Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU DIED" if self.player['health'] <= 0 else "VICTORY!"
            color = self.COLOR_ENEMY if self.player['health'] <= 0 else self.COLOR_GOLD
            
            end_text = self.font_large.render(message, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

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

# Example of how to run the environment
if __name__ == '__main__':
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless
    
    env = GameEnv()
    obs, info = env.reset()
    print("Initial state:")
    print(f"  Score: {info['score']}, Steps: {info['steps']}")

    # Run for a few random steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Score={info['score']}, Terminated={terminated}")
        if terminated:
            print("Episode finished.")
            break
    
    env.close()

    # To play interactively (requires a display)
    # This part will not run with the dummy video driver.
    # To run, comment out the os.environ line above.
    try:
        del os.environ["SDL_VIDEODRIVER"]
        
        env = GameEnv()
        obs, info = env.reset()
        terminated = False
        
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Dungeon Crawler")
        clock = pygame.time.Clock()

        print("\nStarting interactive mode. Close the window to quit.")
        print(GameEnv.user_guide)

        running = True
        while running and not terminated:
            # Action defaults
            movement = 0 # none
            space = 0 # released
            shift = 0 # released
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

            action = [movement, space, shift]
            
            # The game only advances on an action in auto_advance=False mode
            # We will send an action every frame to allow for holding keys
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(10) # Control the speed of the interactive game

        print("Game Over!")
        pygame.time.wait(2000)
        env.close()
        
    except pygame.error as e:
        print(f"\nCould not start interactive mode: {e}")
        print("This is expected if you are in a headless environment.")
        print("To play, run this script in a graphical environment and ensure pygame is installed.")